"""CuPy host side: compile kernel.cu with NVRTC, run on the chosen GPU, launch, copy back.

GPU choice: mytet_cuda.device (`--gpu N` / MYTET_GPU=N, nvidia-smi index, default 0). The index is checked
with nvidia-smi when this module is imported, before CuPy (and so CUDA) is loaded; a bad index raises
device.GpuChoiceError. Only the chosen GPU is made visible to CUDA.
No map or intermediate result is kept between calls (only the compiled modules are reused).
"""
import os
from pathlib import Path

import numpy as np

from . import device

DEVICE = device.bind()  # before CUDA: refuses an index nvidia-smi does not list; CUDA_VISIBLE_DEVICES = its UUID
MIN_FREE_MIB = device.MIN_FREE_MIB  # do not start CUDA on the chosen GPU when it has less free memory than this
os.environ["CUPY_CACHE_IN_MEMORY"] = "1"  # compiled kernels stay in memory
os.environ.setdefault("CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cupy_cache"))

import cupy as cp  # noqa: E402

KERNEL_SOURCE = (Path(__file__).with_name("kernel.cu")).read_text(encoding="utf-8")
NVRTC_OPTIONS = ("--fmad=false", "-std=c++17")
POOL_LIMIT_BYTES = 512 * 1024 * 1024  # device allocations of this process are capped here (context comes on top)
CHUNK_PIXELS = 1 << 19  # pixels per launch: keeps each launch short (the GPU may also drive a display)
THREADS = 256

_modules = {}
_checked = False


def free_mib():
    """Free memory of the chosen GPU now (nvidia-smi)."""
    return device.query(DEVICE["uuid"])["free_mib"]


def _check_device():
    """Runs before any CUDA allocation: free-memory rule, device identity, allocation cap."""
    global _checked
    try:
        now = device.choose()["index"]  # the choice now (--gpu / MYTET_GPU / machine rule; no new nvidia-smi query)
    except device.GpuChoiceError as e:
        raise RuntimeError(f"{e}; this process's CUDA is bound to GPU {DEVICE['index']}; refusing to run") from None
    if now != DEVICE["index"]:  # the choice changed after CUDA was bound to a GPU in this process
        raise RuntimeError(f"the GPU choice is now {now} ({device.GPU_ENV}={os.environ.get(device.GPU_ENV)!r}) but "
                           f"this process's CUDA is bound to GPU {DEVICE['index']}; refusing to run")
    if _checked:
        return
    free = free_mib()
    if free < MIN_FREE_MIB:
        raise RuntimeError(f"GPU {DEVICE['index']} ({DEVICE['name']}) has {free} MiB free (< {MIN_FREE_MIB} MiB); "
                           "not starting GPU work")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != DEVICE["uuid"] or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("CUDA device selection was changed; refusing to run")
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw = props["uuid"]
    h = raw.hex() if isinstance(raw, (bytes, bytearray)) else bytes(raw).hex()
    uuid = f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    if uuid != DEVICE["uuid"]:
        raise RuntimeError(f"wrong GPU {props['name']!r} {uuid}; expected {DEVICE['uuid']}")
    cp.get_default_memory_pool().set_limit(size=POOL_LIMIT_BYTES)
    _checked = True


def device_info():
    from cupy_backends.cuda.libs import nvrtc
    _check_device()
    props = cp.cuda.runtime.getDeviceProperties(0)
    return {"name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "index": DEVICE["index"], "uuid": DEVICE["uuid"], "cc": f"{props['major']}.{props['minor']}",
            "cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "nvrtc": nvrtc.getVersion(), "driver": cp.cuda.runtime.driverGetVersion(),
            "nvrtc_options": list(NVRTC_OPTIONS), "precision": "float64 (complex128)"}


def _module(record=False):
    """Compiled module of kernel.cu (product build, or the record build with -DTET_RECORD)."""
    if record not in _modules:
        _check_device()
        opts = NVRTC_OPTIONS + (("-DTET_RECORD",) if record else ())
        _modules[record] = cp.RawModule(code=KERNEL_SOURCE, options=opts)
    return _modules[record]


def _grid_arrays(x, y):
    w = 1j * y[np.newaxis, :]  # the original's own expression, evaluated by NumPy on the host
    w = w[0]
    return (cp.asarray(np.ascontiguousarray(x, dtype=np.float64)),
            cp.asarray(np.ascontiguousarray(w.real)), cp.asarray(np.ascontiguousarray(w.imag)))


def compute_map(x, y, max_iter, escape_radius, return_iters=False):
    """divergence_map (nx, ny) bool, same layout as the original. Optionally the stop numbers."""
    _check_device()
    nx, ny = len(x), len(y)
    out = np.zeros((nx, ny), dtype=bool)
    iters = np.zeros((nx, ny), dtype=np.int32) if return_iters else None
    if nx == 0 or ny == 0:
        return (out, iters) if return_iters else out
    xd, wrd, wid = _grid_arrays(x, y)
    out_d = cp.empty(nx * ny, dtype=cp.uint8)
    it_d = cp.empty(nx * ny, dtype=cp.int32) if return_iters else None
    k = _module().get_function("tet_map_rows")
    rows = max(1, CHUNK_PIXELS // ny)
    for row0 in range(0, nx, rows):
        nrows = min(rows, nx - row0)
        total = nrows * ny
        k(((total + THREADS - 1) // THREADS,), (THREADS,),
          (xd, wrd, wid, np.int32(ny), np.int32(row0), np.int32(nrows), np.int32(max_iter),
           np.float64(escape_radius), out_d, it_d if it_d is not None else np.uint64(0)))
    out[...] = out_d.get().reshape(nx, ny).view(np.bool_)  # .get() synchronises
    if return_iters:
        iters[...] = it_d.get().reshape(nx, ny)
        return out, iters
    return out


def record(x, y, pixels, max_iter, escape_radius, batch=4096):
    """Trajectories for flat pixel indices i*ny + j: (diverged bool[n], stop int[n], z complex[n, max_iter+1])."""
    _check_device()
    pix = np.ascontiguousarray(pixels, dtype=np.int64)
    n = len(pix)
    div = np.zeros(n, dtype=np.bool_)
    stop = np.zeros(n, dtype=np.int32)
    traj = np.full((n, max_iter + 1), np.nan, dtype=np.complex128)
    if n == 0:
        return div, stop, traj
    xd, wrd, wid = _grid_arrays(x, y)
    k = _module(record=True).get_function("tet_record")
    for s in range(0, n, batch):
        b = pix[s:s + batch]
        m = len(b)
        out_d = cp.empty(m, dtype=cp.uint8)
        it_d = cp.empty(m, dtype=cp.int32)
        traj_d = cp.full(m * 2 * (max_iter + 1), np.nan, dtype=cp.float64)
        k(((m + THREADS - 1) // THREADS,), (THREADS,),
          (xd, wrd, wid, np.int32(len(y)), cp.asarray(b), np.int32(m), np.int32(max_iter),
           np.float64(escape_radius), out_d, it_d, traj_d))
        div[s:s + m] = out_d.get().view(np.bool_)
        stop[s:s + m] = it_d.get()
        traj[s:s + m] = traj_d.get().view(np.complex128).reshape(m, max_iter + 1)
    return div, stop, traj


def onestep(a, b):
    """a ** b elementwise through the kernel's device function."""
    _check_device()
    a = np.ascontiguousarray(a, dtype=np.complex128).ravel()
    b = np.ascontiguousarray(b, dtype=np.complex128).ravel()
    n = len(a)
    out = cp.empty(2 * n, dtype=cp.float64)
    k = _module().get_function("tet_onestep")
    k(((n + THREADS - 1) // THREADS,), (THREADS,),
      (cp.asarray(a.view(np.float64)), cp.asarray(b.view(np.float64)), np.int32(n), out))
    return out.get().view(np.complex128)


def cabs(z):
    _check_device()
    z = np.ascontiguousarray(z, dtype=np.complex128).ravel()
    n = len(z)
    out = cp.empty(n, dtype=cp.float64)
    k = _module().get_function("tet_abs")
    k(((n + THREADS - 1) // THREADS,), (THREADS,), (cp.asarray(z.view(np.float64)), np.int32(n), out))
    return out.get()


def grid_c(x, y):
    """c for every pixel as the kernel forms it."""
    _check_device()
    nx, ny = len(x), len(y)
    xd, wrd, wid = _grid_arrays(x, y)
    out = cp.empty(2 * nx * ny, dtype=cp.float64)
    k = _module().get_function("tet_grid")
    k(((nx * ny + THREADS - 1) // THREADS,), (THREADS,),
      (xd, wrd, wid, np.int32(ny), np.int32(0), np.int32(nx), out))
    return out.get().view(np.complex128).reshape(nx, ny)
