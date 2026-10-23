"""CPU planning and event-polled CUDA jobs for the rebasing perturbation engine."""
from dataclasses import dataclass
from pathlib import Path
import copy
import time
import numpy as np
from . import hp, limits, reference

SOURCE = Path(__file__).with_name('perturb_kernel.cu').read_text(encoding='utf-8')
_modules = {}
SELF_MAX = 64


def _module(test=False):
    from mytet_cuda import gpu
    gpu._check_device()
    if test not in _modules:
        _modules[test] = gpu.cp.RawModule(code=SOURCE, options=gpu.NVRTC_OPTIONS + (('-DPT_TEST',) if test else ()))
    return _modules[test]


@dataclass
class Plan:
    x0: object
    y0: object
    eps: float
    ref: object
    gx: object
    gy: object
    info: dict
    seconds: float


def plan(x0, y0, eps, *, max_iter=None, ref_digits=None, ref_pixel=None):
    """Prepare all CPU data without importing CuPy (also for a rejected view)."""
    start = time.perf_counter()
    x0, y0 = hp.D(x0), hp.D(y0)
    n = limits.max_iter_for(eps) if max_iter is None else int(max_iter)
    digits = limits.ref_digits_for(eps) if ref_digits is None else int(ref_digits)
    if n<0 or digits<2:
        raise ValueError('invalid iteration count or reference precision')
    if ref_pixel is not None and (len(ref_pixel)!=2 or any(int(v)!=v for v in ref_pixel)
                                 or not (0<=ref_pixel[0]<1920 and 0<=ref_pixel[1]<1080)):
        raise ValueError('reference pixel outside view')
    info = {'path':'perturb', 'max_iter':n, 'ref_digits':digits, 'refs':[],
            'flag_counts':{}, 'out_of_range':False, 'reason':None, 'unique_x':0, 'unique_y':0,
            'kernel_passes':0, 'rebases':0, 'cycle_exits':0, 'fallback_pixels':0,
            't_ms':{'reference':0., 'offsets':0., 'kernel':0., 'fallback':0., 'total':0.}}
    ref, gx, gy = None, None, None
    reason = limits.view_reason(x0,y0,eps)
    if reason:
        info.update(out_of_range=True, reason=reason)
    else:
        ref, seconds, tried = reference.choose(x0,y0,eps,n,digits,None if ref_pixel is None else [tuple(ref_pixel)])
        info['t_ms']['reference'] = seconds*1000
        begin = time.perf_counter()
        gx, gy = reference.offsets(x0,y0,eps,ref)
        info['t_ms']['offsets'] = 1000*(time.perf_counter()-begin)
        info['unique_x'], info['unique_y'] = int(np.unique(gx).size), int(np.unique(gy).size)
        info['refs'] = [{'pixel':list(ref.pixel) if ref.pixel else None, 'center':list(ref.center),
                         'n_ref':ref.n_ref, 'min_log10_abs':ref.min_log10_abs,
                         'max_log10_abs':ref.max_log10_abs, 'orbit_s':ref.seconds,
                         'candidates':tried, 'cycle':ref.cycle,
                         'unique_x':info['unique_x'], 'unique_y':info['unique_y']}]
        issue = reference.range_issue(ref,n)
        if issue:
            info.update(out_of_range=True, reason='reference_exponent', reference_exponent=issue)
    seconds = time.perf_counter()-start
    info['t_ms']['total'] = seconds*1000
    return Plan(x0,y0,float(eps),ref,gx,gy,info,seconds)


class Job:
    def __init__(self, prepared, stride, stream, pixels=None):
        self.plan, self.stride = prepared, stride
        self.pixels = pixels
        self.shape = ((1920+stride-1)//stride, (1080+stride-1)//stride) if pixels is None else (len(pixels),)
        self.count = int(np.prod(self.shape))
        self.info = copy.deepcopy(prepared.info)
        self.started = time.perf_counter()
        self._result = None
        self.end = None
        self.buffers = []  # retain every input until the completion event
        if self.info['out_of_range'] or not self.count:
            return
        from mytet_cuda import gpu
        gpu._check_device()
        self.cp = cp = gpu.cp
        self.stream = cp.cuda.get_current_stream() if stream is None else stream
        ref = prepared.ref
        with self.stream:
            def device(array):
                value = cp.asarray(array)
                self.buffers.append(value)
                return value
            self.out, self.flags, self.stops, self.rebases, self.cycles = (
                cp.empty(self.count,dtype=d) for d in (cp.uint8,cp.uint8,cp.int32,cp.int32,cp.uint8))
            args = (device(prepared.gx),device(prepared.gy),np.float64(ref.center[0]),np.float64(ref.center[1]),
                    np.float64(ref.lc.real),np.float64(ref.lc.imag),device(ref.z),device(ref.log_abs),
                    device(ref.log_abs_lo),device(ref.arg),device(ref.arg_lo),device(ref.margin),
                    np.int32(len(ref.z)),np.int32(self.info['max_iter']),
                    np.uint64(0) if pixels is None else device(pixels),np.int32(stride),
                    np.int32(self.shape[1] if pixels is None else 1080))
            kernel = _module().get_function('pt_map')
            self.begin, self.end = cp.cuda.Event(), cp.cuda.Event()
            self.begin.record(self.stream)
            # Bound launches, including sparse sets; no event synchronization here.
            chunk = min(gpu.CHUNK_PIXELS, 1<<17)
            for offset in range(0,self.count,chunk):
                count = min(chunk,self.count-offset)
                kernel(((count+gpu.THREADS-1)//gpu.THREADS,),(gpu.THREADS,),
                       args+(np.int32(offset),np.int32(count),self.out,self.flags,self.stops,self.rebases,self.cycles),
                       stream=self.stream)
                self.info['kernel_passes'] += 1
            self.end.record(self.stream)

    def done(self):
        """Only cudaEventQuery, never synchronization or a host transfer."""
        return self.end is None or bool(self.end.done)  # CuPy Event exposes .done (cudaEventQuery), not .query()

    def result(self, *, return_stop=False):
        """Wait if necessary; direct high precision self references live here."""
        if self._result is None:
            p, info = self.plan, self.info
            if self.end is None:
                out = np.zeros(self.count,dtype=bool)
                flags = np.zeros(self.count,dtype=np.uint8)
                stops = np.zeros(self.count,dtype=np.int32)
            else:
                self.end.synchronize()
                info['t_ms']['kernel'] = float(self.cp.cuda.get_elapsed_time(self.begin,self.end))
                out = self.out.get(stream=self.stream).view(np.bool_)
                flags = self.flags.get(stream=self.stream)
                stops = self.stops.get(stream=self.stream)
                info['rebases'] = int(self.rebases.get(stream=self.stream).sum(dtype=np.int64))
                info['cycle_exits'] = int(self.cycles.get(stream=self.stream).sum())
                bad = np.flatnonzero(flags==3)
                if 0<len(bad)<=SELF_MAX:
                    begin = time.perf_counter()
                    for index in bad:
                        flat = (int(self.pixels[index]) if self.pixels is not None else
                                (int(index)//self.shape[1]*self.stride)*1080+int(index)%self.shape[1]*self.stride)
                        pixel = (flat//1080,flat%1080)
                        own = reference.at_pixel(p.x0,p.y0,p.eps,pixel,info['max_iter'],info['ref_digits'])
                        if reference.range_issue(own,info['max_iter']) is None:
                            out[index] = own.n_ref is not None
                            stops[index] = own.n_ref if own.n_ref is not None else info['max_iter']
                            flags[index] = 0
                            info['fallback_pixels'] += 1
                            info['refs'].append({'pixel':list(pixel),'center':list(own.center),
                                                 'n_ref':own.n_ref,'self_reference':True,'orbit_s':own.seconds})
                    info['t_ms']['fallback'] = 1000*(time.perf_counter()-begin)
                if np.any(flags==2):
                    info.update(out_of_range=True,reason='pixel_exponent')
                elif np.any(flags==3):
                    info.update(out_of_range=True,reason='unresolved')
            info['flag_counts'] = {str(i):int(np.count_nonzero(flags==i)) for i in range(4)}
            info['t_ms']['total'] = 1000*(p.seconds+time.perf_counter()-self.started)
            self._result = out.reshape(self.shape), stops.reshape(self.shape)
        out, stops = self._result
        info = copy.deepcopy(self.info)
        if return_stop or self.pixels is not None:
            info['stop'] = stops.tolist() if self.pixels is not None else stops.copy()
        return out, info


def launch(plan, *, stride=1, stream=None):
    if isinstance(stride,bool) or int(stride)!=stride or stride<1:
        raise ValueError('stride must be a positive integer')
    return Job(plan,int(stride),stream)


def deep_map(x0,y0,eps, *, max_iter=None, ref_digits=None, ref_pixel=None, timing=True, return_stop=False):
    prepared = plan(x0,y0,eps,max_iter=max_iter,ref_digits=ref_digits,ref_pixel=ref_pixel)
    out, info = launch(prepared).result(return_stop=return_stop)
    if not timing:
        info['t_ms']['kernel'] = None
    return out, info


def deep_samples(x0,y0,eps,pixels, **kw):
    pairs = np.asarray(pixels,dtype=np.int64)
    if not pairs.size:
        pairs = pairs.reshape(0,2)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0) or np.any(pairs>=[1920,1080]):
        raise ValueError('pixels must be in-bounds (px,py) pairs')
    timing = kw.pop('timing',True)
    kw.pop('return_stop',False)  # sparse API always retains its stage-one stop list
    prepared = plan(x0,y0,eps,**kw)
    job = Job(prepared,1,None,np.ascontiguousarray(pairs[:,0]*1080+pairs[:,1]))
    out, info = job.result(return_stop=True)
    if not timing:
        info['t_ms']['kernel'] = None
    return out, info
