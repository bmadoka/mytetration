"""CPU-only high precision view transforms and independent timed input playback."""
import queue
import threading
import time

import numpy as np
from deep import hp


def state_key(state):
    def decimal_key(value):
        # Decimal.normalize() uses the ambient context and can round HP centres.
        value = hp.D(value)
        if value.is_zero():
            return '0'
        sign, digits, exponent = value.as_tuple()
        while digits[-1] == 0:
            digits, exponent = digits[:-1], exponent + 1
        return str(hp.D((sign, digits, exponent)))
    return decimal_key(state[0]), decimal_key(state[1]), float(state[2]).hex()


def warp_indices(display, source, stride=1):
    """Nearest full-grid indices, edge-clamped like the shallow hybrid, then //stride.

    Centre subtraction is Decimal BEFORE conversion to float. Extent differences
    are also Decimal. Coarse samples are anchored at full-grid index 0.
    """
    ds, ss = hp.spacing(display[2]), hp.spacing(source[2])
    de, se = hp.extents(display[2]), hp.extents(source[2])
    offsets = [float(hp.CTX.divide(hp.CTX.add(hp.CTX.subtract(d, s), hp.CTX.subtract(b, a)), step))
               for d, s, a, b, step in zip(display[:2], source[:2], de, se, ss)]
    scales = [float(hp.CTX.divide(d, s)) for d, s in zip(ds, ss)]
    axes = [np.clip(np.floor(offset + np.arange(n) * scale + .5), 0, n-1).astype(np.int32) // stride
            for n, offset, scale in zip((1920, 1080), offsets, scales)]
    return *axes, dict(scale=scales[0], offset=offsets, stride=stride, edge='clamp')


def warp_map(m, display, source, stride=1):
    ix, iy, transform = warp_indices(display, source, stride)
    return m[np.ix_(ix, iy)], transform


class ThreadedReplay:
    """An input producer that never depends on a render-loop tick (no OS input)."""
    def __init__(self, events, t0, *, clock=time.perf_counter):
        self.events, self.t0, self.clock = events, t0, clock
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name='script-playback', daemon=True)

    def _run(self):
        for k, ev in enumerate(self.events):
            scheduled = self.t0 + ev['t']
            if self.stop.wait(max(0., scheduled - self.clock())):
                break
            self.queue.put(dict(ev, input_index=k, t_scheduled=scheduled, t_enqueued=self.clock()))

    def start(self):
        self.thread.start()
        return self

    def drain(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except queue.Empty:
                return

    def close(self):
        self.stop.set()
        self.thread.join(1.)


def timing_metrics(inputs, frames, begin, end):
    """Each input uses the FIRST present reflecting its index, even when batched."""
    latencies = []
    missing = []
    for k, ev in enumerate(inputs):
        first = next((f['t_present'] for f in frames if f['last_input'] >= k), None)
        if first is None:
            missing.append(k)
        else:
            latencies.append(1000 * (first - ev['t_scheduled']))
    # Include both neighbours of every interval intersecting the measurement window.
    gaps = [1000*(b['t_present']-a['t_present']) for a, b in zip(frames, frames[1:])
            if b['t_present'] >= begin and a['t_present'] < end]

    def distribution(values, percentiles):
        return dict(n=len(values), values_ms=values, max=max(values) if values else None,
                    **{f'p{p}': float(np.percentile(values, p)) if values else None for p in percentiles})
    return dict(input_latency_ms=distribution(latencies, (50, 95)),
                present_interval_ms=distribution(gaps, (50, 99)), missing_inputs=missing,
                measurement_begin=begin, measurement_end=end, percentile='numpy linear')
