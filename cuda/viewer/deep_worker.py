"""Spawn-only deep computation. Importing this module does not import CUDA.

The parent owns a single replaceable pending request. Only one request is sent
to the child at a time; obsolete pending work never reaches plan/launch.
The engine contract has no cancellation method: an already running job finishes
in the child and is cached, without blocking the renderer.
"""
import hashlib
from collections import OrderedDict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback
import zipfile

import numpy as np
from deep import hp, limits
from viewer.deep_view import state_key


def remember(cache, key, value, capacity):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > capacity:
        cache.popitem(last=False)


def gpu_engine(directory, responses):
    from mytet_cuda import gpu
    from scripts.perturb_tools import audit_gpu, require_memory
    require_memory()
    device = gpu.device_info()  # chosen-GPU, memory and allocation-limit rules
    gpu.cp.empty(1)             # create the context, then immediately audit THIS pid
    audit = audit_gpu('deep worker context', directory)
    responses.put(dict(event='audit', audit=audit, device=device, t=time.perf_counter()))
    from deep import perturb
    return perturb


def worker_main(requests, responses, directory, engine_factory=None):
    engine, plans, computed = None, OrderedDict(), set()
    try:
        while True:
            request = requests.get()
            if request is None:
                break
            key = state_key(hp.from_strings(request['x0'], request['y0'],
                                            float.fromhex(request['eps_hex'])))
            unit = (*key, request['stride'])
            if unit in computed and not request.get('recompute_after_eviction'):
                responses.put(dict(event='duplicate_rejected', id=request['id'], unit=unit,
                                   t=time.perf_counter()))
                continue
            if engine is None:
                engine = (engine_factory or gpu_engine)(directory, responses)
            started = time.perf_counter()
            responses.put(dict(event='start', **request, t_start=started))
            computed.add(unit)
            if key not in plans:
                remember(plans, key, engine.plan(request['x0'], request['y0'], float.fromhex(request['eps_hex']),
                                                max_iter=request['max_iter']), 8)
            plans.move_to_end(key)
            job = engine.launch(plans[key], stride=request['stride'])
            while not job.done():
                time.sleep(.001)
            m, info = job.result()  # device sync / CPU fallback is confined to this child
            ended = time.perf_counter()
            shape = ((1920+request['stride']-1)//request['stride'],
                     (1080+request['stride']-1)//request['stride'])
            m = np.asarray(m, dtype=bool)
            if m.shape != shape:
                raise ValueError(f'engine map shape {m.shape} != {shape}')
            responses.put(dict(event='end', **request, bits=np.packbits(m).tobytes(),
                               shape=shape, info=info, map_sha256=hashlib.sha256(m.tobytes()).hexdigest(),
                               t_start=started, t_end=ended))
    except BaseException:
        responses.put(dict(event='error', error=traceback.format_exc(), t=time.perf_counter()))


class ProcessWorker:
    def __init__(self, directory, engine_factory=None):
        ctx = mp.get_context('spawn')
        self.requests, self.responses = ctx.Queue(), ctx.Queue()
        self.process = ctx.Process(target=worker_main,
                                   args=(self.requests, self.responses, str(directory), engine_factory),
                                   name='deep-compute', daemon=True)
        self.process.start()

    def submit(self, request):
        self.requests.put_nowait(request)

    def poll(self):
        while True:
            try:
                yield self.responses.get_nowait()
            except queue.Empty:
                if self.process.exitcode not in (None, 0):
                    raise RuntimeError(f'deep worker exited {self.process.exitcode}')
                return

    def close(self):
        self.requests.put_nowait(None)
        self.process.join(.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2.)
        self.requests.close()
        self.responses.close()


class DeepController:
    """Pure scheduling policy around an injectable process transport."""
    def __init__(self, worker, directory=None):
        self.worker, self.directory = worker, Path(directory) if directory else None
        self.active = self.pending = self.source = None
        self.cache, self.records, self.finals = OrderedDict(), [], OrderedDict()
        self.coarse = OrderedDict()  # packed previews also have a finite retention window
        self.computed = set()
        self.archived_ids = set()
        self.next_id = 0
        self.desired = None

    def log(self, record):
        row = {k: v for k, v in record.items() if k not in ('bits', 'map')}
        row.setdefault('t', time.perf_counter())
        self.records.append(row)
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory/'deep_jobs.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')

    def poll(self):
        for response in self.worker.poll():
            response['t_received'] = time.perf_counter()
            self.log(response)
            if response['event'] == 'error':
                raise RuntimeError(response['error'])
            if response['event'] == 'duplicate_rejected':
                raise RuntimeError('deep worker rejected duplicate computation')
            if response['event'] != 'end':
                continue
            key = state_key(hp.from_strings(response['x0'], response['y0'],
                                            float.fromhex(response['eps_hex'])))
            response['hp'] = hp.from_strings(*key[:2], float.fromhex(key[2]))
            unit = (*key, response['stride'])
            self.computed.add(unit)
            if response['stride'] == 1:
                remember(self.finals, unit, response.copy(), 256)
                if self.directory:
                    self._archive_final(response, self.directory)
            else:
                remember(self.coarse, unit, response.copy(), 256)
            self.source = self._decode(unit, response)
            self.active = None

    def _decode(self, unit, packed):
        response = packed.copy()
        n = int(np.prod(response['shape']))
        response['map'] = np.unpackbits(np.frombuffer(response['bits'], np.uint8), count=n).reshape(response['shape']).astype(bool)
        remember(self.cache, unit, response, 16)
        return response

    def _cached(self, unit):
        packed = self.finals if unit[-1] == 1 else self.coarse
        if unit in self.cache:
            self.cache.move_to_end(unit)
            if unit in packed:
                packed.move_to_end(unit)
            return self.cache[unit]
        if unit in packed:
            packed.move_to_end(unit)
            return self._decode(unit, packed[unit])
        return None

    def _archive_final(self, response, directory):
        if response['id'] in self.archived_ids:
            return
        directory.mkdir(parents=True, exist_ok=True)
        # A closed zip member is a valid npy array; no previous arrays enter RAM.
        with zipfile.ZipFile(directory/'deep_final_maps.npz', 'a') as archive:
            with archive.open(str(response['id'])+'.npy', 'w') as stream:
                np.lib.format.write_array(stream, np.frombuffer(response['bits'], np.uint8), allow_pickle=False)
        row = {k:v for k,v in response.items() if k not in ('bits', 'map', 'hp')}
        with (directory/'deep_final_maps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        self.archived_ids.add(response['id'])

    def save_final_archive(self, directory):
        directory = Path(directory)
        if self.directory is not None and directory != self.directory:
            raise ValueError('final archive directory changed during playback')
        for response in self.finals.values():
            self._archive_final(response, directory)

    def tick(self, high, moving, last_input=-1):
        self.poll()
        key = state_key(high)
        self.desired = key
        full = self._cached((*key, 1))
        coarse = None if full else self._cached((*key, 8))
        wanted = None if full else (8 if coarse is None else (None if moving else 1))
        unit = (*key, wanted) if wanted else None
        if self.pending and (unit is None or self._unit(self.pending) != unit):
            self.log(dict(event='cancel', **self.pending, reason='superseded_before_start'))
            self.pending = None
        if wanted and (self.active is None or self._unit(self.active) != unit) and self.pending is None:
            self.pending = dict(id=self.next_id, x0=key[0], y0=key[1], eps=high[2], eps_hex=key[2],
                                stride=wanted, max_iter=limits.max_iter_for(high[2]), last_input=last_input)
            self.next_id += 1
            if unit in self.computed:
                self.pending['recompute_after_eviction'] = True
            self.log(dict(event='request', **self.pending))
        if self.active is None and self.pending:
            self.active, self.pending = self.pending, None
            if self.active.get('recompute_after_eviction'):
                self.log(dict(event='recompute_after_eviction', **self.active))
            self.worker.submit(self.active)
        # Prefer an exact cached map, otherwise use the most recent received one.
        return full or coarse or self.source

    @staticmethod
    def _unit(request):
        return (*state_key(hp.from_strings(request['x0'], request['y0'],
                                           float.fromhex(request['eps_hex']))), request['stride'])

    def final_for(self, high):
        return self._cached((*state_key(high), 1))

    def suspend(self):
        """Leaving the deep display: retain results, discard unsent work."""
        self.poll()
        if self.pending:
            self.log(dict(event='cancel', **self.pending, reason='shallow_before_start'))
            self.pending = None

    def close(self):
        if self.pending:
            self.log(dict(event='cancel', **self.pending, reason='viewer_closed_before_start'))
            self.pending = None
        if self.active:
            self.log(dict(event='cancel', **self.active, reason='viewer_closed_inflight_process_terminated'))
        self.worker.close()
