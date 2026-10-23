"""Full 1080p fast maps with measured f32/f64 selection; no map/orbit reuse between frames.

GPU selection and allocation limits are enforced by mytet_cuda.gpu (the chosen GPU).
Only the compiled kernels are cached. Coordinate output comes from the map kernel,
at the point where c is passed to the orbit calculation.
"""
import os
from pathlib import Path

import numpy as np

import mytet_cuda
from mytet_cuda import gpu, variants

import params

cp = gpu.cp
SOURCE = Path(__file__).with_name("fast_kernel.cu").read_text(encoding="utf-8")
OPTIONS = ("-std=c++17",)
F64_OPTIONS = OPTIONS + ("--fmad=false",)
_modules = {}
calls = {"fast": 0, "precise": 0, "f32": 0, "f64": 0, "precision": []}

# Relative x-grid spacing, calibrated with a separate precision bench (block share and mismatch per notch).
F32_MIN_REL_SPACING = 1e-5
F32_MIN_SCALE = 0.1
USE_CYCLE_EXIT = True
F32_C_MAX = 60.0  # absolute view extent; measured D1 drag counterexamples reach ~95

# Measurement override only; the application uses this one row-count constant.
ROWS_PER_LAUNCH = int(os.environ.get("MYTET_CHUNK_ROWS", "32"))
if ROWS_PER_LAUNCH < 1:
    raise ValueError("MYTET_CHUNK_ROWS must be positive")


def relative_spacing(x0, y0, eps):
    return (2 * eps / 1919) / max(abs(x0), abs(y0), 1e-300)


def choose_precision(x0, y0, eps):
    """Measured f32 spacing, scale, extent and coordinate guards; f64 otherwise."""
    if (relative_spacing(x0, y0, eps) < F32_MIN_REL_SPACING
            or max(abs(x0), abs(y0), eps) < F32_MIN_SCALE
            or max(abs(x0), abs(y0)) + eps > F32_C_MAX):
        return "f64"
    x, y, _ = variants.grid(params.VARIANT, x0, y0, eps, params.NX)
    xf, yf = x.astype(np.float32), y.astype(np.float32)
    safe = (np.isfinite(xf).all() and np.isfinite(yf).all()
            and np.all(xf[1:] != xf[:-1]) and np.all(yf[1:] != yf[:-1]))
    return "f32" if safe else "f64"


def _kernel(precision):
    if precision not in _modules:
        gpu._check_device()
        options = OPTIONS if precision == "f32" else F64_OPTIONS
        _modules[precision] = cp.RawModule(code=SOURCE, options=options)
    return _modules[precision].get_function(precision + "_map_rows")


def fast_map(x0, y0, eps, *, precision=None, return_coordinates=False, timing=None, cycle_exit=None):
    """Fresh map. Override is for bench/controls only, never the viewer.

    return_coordinates: (map, c[nx,ny,2]) read from the actual orbit kernel.
    timing: optional dict receiving CUDA-event time of all row launches (ms);
    excludes grid upload, allocation and map readback. No CUDA timing events in viewer frames.
    """
    # G2_FAULT_TARGET: plain f32 orbit with double coordinates, never perturbation.
    precision = choose_precision(x0, y0, eps) if precision is None else precision
    if precision not in ("f32", "f64"):
        raise ValueError(precision)
    calls["fast"] += 1
    calls[precision] += 1
    calls["precision"].append(precision)
    k = _kernel(precision)
    cycle_exit = USE_CYCLE_EXIT if cycle_exit is None else cycle_exit
    dtype = np.float32 if precision == "f32" else np.float64
    x, y, _ = variants.grid(params.VARIANT, x0, y0, eps, params.NX)
    nx, ny = len(x), len(y)
    w = (1j * y[np.newaxis, :])[0]
    xd = cp.asarray(np.ascontiguousarray(x, dtype=dtype))
    wrd = cp.asarray(np.ascontiguousarray(w.real, dtype=dtype))
    wid = cp.asarray(np.ascontiguousarray(w.imag, dtype=dtype))
    out_d = cp.empty(nx * ny, dtype=cp.uint8)
    coord_d = cp.empty((nx, ny, 2), dtype=dtype) if return_coordinates else None
    if timing is not None:
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
    rows = max(1, gpu.CHUNK_PIXELS // ny)
    for row0 in range(0, nx, rows):
        nrows = min(rows, nx - row0)
        total = nrows * ny
        args = (xd, wrd, wid, np.int32(ny), np.int32(row0), np.int32(nrows), np.int32(params.MAX_ITER),
                dtype(params.ESCAPE_RADIUS), out_d, coord_d if coord_d is not None else np.uint64(0))
        if precision == "f64":
            args += (np.int32(bool(cycle_exit)),)
        k(((total + gpu.THREADS - 1) // gpu.THREADS,), (gpu.THREADS,), args)
    if timing is not None:
        end.record()
        end.synchronize()
        timing["gpu_ms"] = float(cp.cuda.get_elapsed_time(begin, end))
    out = out_d.get().reshape(nx, ny).view(np.bool_)
    return (out, coord_d.get()) if return_coordinates else out



def precise_map(x0, y0, eps):
    calls["precise"] += 1
    return mytet_cuda.divergence_map(params.VARIANT, x0, y0, eps, params.NX, params.MAX_ITER, params.ESCAPE_RADIUS)


def pool_bytes():
    return cp.get_default_memory_pool().total_bytes()


_stream_priorities = None


def stream_priority_range():
    """CUDA driver query missing from this installed CuPy runtime wrapper."""
    global _stream_priorities
    if _stream_priorities is not None:
        return _stream_priorities
    import ctypes
    gpu._check_device()
    cp.cuda.runtime.free(0)  # create the pinned GPU's primary context before driver query
    driver=ctypes.WinDLL("nvcuda.dll")
    query=driver.cuCtxGetStreamPriorityRange
    query.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
    query.restype=ctypes.c_int
    low,high=ctypes.c_int(),ctypes.c_int()
    status=query(ctypes.byref(low),ctypes.byref(high))
    if status:
        raise RuntimeError(f"cuCtxGetStreamPriorityRange failed: {status}")
    _stream_priorities = low.value,high.value
    return _stream_priorities


STREAMS = int(os.environ.get("MYTET_STREAMS", "4"))
MX, MY = map(int, os.environ.get("MYTET_MARGIN", "480,270").replace("x", ",").split(",")) if os.environ.get("MYTET_MARGIN", "") != "0" else (0, 0)
if STREAMS < 1 or min(MX, MY) < 0:
    raise ValueError("invalid stream count or margin")


def extended_axes(x0, y0, eps, mx=MX, my=MY):
    """linspace operation order, extended integer indices; exact screen endpoints."""
    axes = []
    for center, extent, n, margin in ((x0, eps, 1920, mx), (y0, eps*(9/16), 1080, my)):
        start, stop = center-extent, center+extent
        axis = np.arange(-margin, n+margin, dtype=np.float64) * ((stop-start)/(n-1))
        axis += start
        axis[margin+n-1] = stop
        axes.append(axis)
    return axes


def margin_rectangles(mx=MX, my=MY):
    """Disjoint extended-array rectangles; never compute a screen pixel."""
    nx, ny = 1920+2*mx, 1080+2*my
    rects = [(0,mx,0,ny), (mx+1920,nx,0,ny), (mx,mx+1920,0,my), (mx,mx+1920,my+1080,ny)]
    return [(x, min(x+ROWS_PER_LAUNCH,x1), y0, y1) for x0,x1,y0,y1 in rects
            if y1>y0 for x in range(x0,x1,ROWS_PER_LAUNCH)]


class MarginChunk:
    """One ring rectangle on a lowest-priority stream, including pinned download."""
    def __init__(self, p, precision, rect, mx=MX, my=MY):
        gpu._check_device()
        self.rect = rect
        x0,x1,y0,y1 = rect
        x,y = extended_axes(*p, mx, my)
        dtype = np.float32 if precision == "f32" else np.float64
        w=1j*y[y0:y1]  # preserve the screen kernel's signed-zero complex multiplication
        self.host_axes = [np.ascontiguousarray(a, dtype=dtype) for a in (x[x0:x1], w.real, w.imag)]
        low, _ = stream_priority_range()
        self.stream = cp.cuda.Stream(non_blocking=True, priority=low)
        self.event = cp.cuda.Event()
        self.pinned = cp.cuda.alloc_pinned_memory((x1-x0)*(y1-y0))
        self.host = np.frombuffer(self.pinned,dtype=np.uint8,count=(x1-x0)*(y1-y0)).reshape(x1-x0,y1-y0)
        with self.stream:
            self.axes = [cp.asarray(a) for a in self.host_axes]
            self.out = cp.empty(self.host.shape,dtype=cp.uint8)
            args = (*self.axes,np.int32(y1-y0),np.int32(0),np.int32(x1-x0),np.int32(params.MAX_ITER),
                    dtype(params.ESCAPE_RADIUS),self.out,np.uint64(0))
            if precision == "f64":
                args += (np.int32(bool(USE_CYCLE_EXIT)),)
            total=self.host.size
            _kernel(precision)(((total+gpu.THREADS-1)//gpu.THREADS,),(gpu.THREADS,),args)
            self.out.get(out=self.host,stream=self.stream,blocking=False)
            self.event.record(self.stream)

    def poll(self):
        return self.host.view(np.bool_) if self.event.done else None


class FreshJob:
    """Full fast_map on a nonblocking stream; poll never waits for CUDA.

    All row chunks are queued at start. A completion event follows an async
    download to pinned host memory, so poll only copies CPU bytes once ready.
    The buffers remain owned by the job through the event. ROWS_PER_LAUNCH
    targets <=8ms on the chosen GPU; the caller presents while these run.
    """
    def __init__(self, x0, y0, eps, *, timing=None):
        import time
        self.params = tuple(float(v) for v in (x0, y0, eps))
        self.t_start = time.perf_counter()
        self.precision = choose_precision(*self.params)
        gpu._check_device()
        k = _kernel(self.precision)
        dtype = np.float32 if self.precision == "f32" else np.float64
        x, y, _ = variants.grid(params.VARIANT, *self.params, params.NX)
        self.nx, self.ny = len(x), len(y)
        w = (1j * y[np.newaxis, :])[0]
        self.host_axes = [np.ascontiguousarray(a, dtype=dtype) for a in (x, w.real, w.imag)]
        _, high = stream_priority_range()
        self.streams = [cp.cuda.Stream(non_blocking=True, priority=high) for _ in range(STREAMS)]
        self.stream = self.streams[0]
        self.event = cp.cuda.Event()
        self.pinned = cp.cuda.alloc_pinned_memory(self.nx * self.ny)
        self.host = np.frombuffer(self.pinned, dtype=np.uint8, count=self.nx * self.ny).reshape(self.nx, self.ny)
        self.result = None
        self.timing = timing
        self.launch_events = []
        calls["fast"] += 1
        calls[self.precision] += 1
        calls["precision"].append(self.precision)
        with self.stream:
            self.axes = [cp.asarray(a) for a in self.host_axes]
            self.out = cp.empty(self.nx * self.ny, dtype=cp.uint8)
            ready = cp.cuda.Event()
            ready.record(self.stream)
        for stream in self.streams[1:]:
            stream.wait_event(ready)
        for number, row0 in enumerate(range(0, self.nx, ROWS_PER_LAUNCH)):
            stream = self.streams[number % STREAMS]
            with stream:
                nrows = min(ROWS_PER_LAUNCH, self.nx - row0)
                total = nrows * self.ny
                args = (*self.axes, np.int32(self.ny), np.int32(row0), np.int32(nrows),
                        np.int32(params.MAX_ITER), dtype(params.ESCAPE_RADIUS), self.out, np.uint64(0))
                if self.precision == "f64":
                    args += (np.int32(bool(USE_CYCLE_EXIT)),)
                if timing is not None:
                    begin, end = cp.cuda.Event(), cp.cuda.Event()
                    begin.record(stream)
                k(((total + gpu.THREADS - 1) // gpu.THREADS,), (gpu.THREADS,), args)
                if timing is not None:
                    end.record(stream)
                    self.launch_events.append((begin, end))
        self.join_events = []
        for stream in self.streams:
            end = cp.cuda.Event()
            end.record(stream)
            self.join_events.append(end)
        with self.stream:
            for end in self.join_events:
                self.stream.wait_event(end)
            self.out.reshape(self.nx, self.ny).get(out=self.host, stream=self.stream, blocking=False)
            self.event.record(self.stream)

    def poll(self):
        """None until complete, then the same bool bytes as fast_map(*params)."""
        if self.result is None and self.event.done:
            self.result = self.host.view(np.bool_).copy()
            self.result.setflags(write=False)
            if self.timing is not None:
                self.timing["launch_ms"] = [float(cp.cuda.get_elapsed_time(a, b)) for a, b in self.launch_events]
        return self.result
