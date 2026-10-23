"""Bench-only perturbation experiment; original arithmetic/options retained for comparison.

No viewer import or compile path depends on this module. GPU access uses fast.gpu's chosen-GPU pin.
"""
from pathlib import Path

import fast
import numpy as np
from fast import cp, gpu, variants, params

SOURCE = Path(__file__).with_name("deepzoom_pt32_kernel.cu").read_text(encoding="utf-8")
OPTIONS = fast.OPTIONS
PT32_RHO = 0.05
PT32_REFERENCE_RADIUS = 4
USE_CYCLE_EXIT = True
_mod = None


def _kernel(precision):
    global _mod
    if _mod is None:
        gpu._check_device()
        _mod = cp.RawModule(code=SOURCE, options=OPTIONS)
    return _mod.get_function(precision + "_map_rows")


def _reference_candidates(nx, ny, radius):
    """Deterministic host search order, starting at a grid point by the centre."""
    ir, jr = nx // 2, ny // 2
    points = [(ir, jr)]
    for r in range(1, radius + 1):
        points.extend((ir + di, jr + dj) for di in range(-r, r + 1)
                      for dj in range(-r, r + 1) if max(abs(di), abs(dj)) == r
                      and 0 <= ir + di < nx and 0 <= jr + dj < ny)
    return np.asarray(points, dtype=np.int32)


def fast_map(x0, y0, eps, *, return_coordinates=False, timing=None, cycle_exit=None,
             diagnostics=None, rho=None, reference_radius=None):
    k = _kernel("pt32")  # checks device/free memory before any allocation
    rho = PT32_RHO if rho is None else float(rho)
    radius = PT32_REFERENCE_RADIUS if reference_radius is None else int(reference_radius)
    if not 0 < rho <= 1 or not 0 <= radius <= 16:
        raise ValueError("pt32 requires 0 < rho <= 1 and 0 <= reference_radius <= 16")
    cycle_exit = USE_CYCLE_EXIT if cycle_exit is None else cycle_exit
    x, y, _ = variants.grid(params.VARIANT, x0, y0, eps, params.NX)
    nx, ny = len(x), len(y)
    xd, yd = cp.asarray(x), cp.asarray(y)
    wrd = cp.asarray((1j*y).real)
    candidates = _reference_candidates(nx, ny, radius)
    candidates_d = cp.asarray(candidates)
    orbit_d = cp.empty((len(candidates), params.MAX_ITER + 2, 2), dtype=cp.float64)
    stops_d = cp.empty(len(candidates), dtype=cp.int32)
    out_d = cp.empty(nx * ny, dtype=cp.uint8)
    coord_d = cp.empty((nx, ny, 2), dtype=cp.float32) if return_coordinates else None
    switch_d = cp.empty(nx * ny, dtype=cp.int32) if diagnostics is not None else None
    if timing is not None:
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
    _mod.get_function("pt32_references")(((len(candidates)+127)//128,), (128,),
        (xd, yd, candidates_d, np.int32(len(candidates)), np.int32(params.MAX_ITER),
         np.float64(params.ESCAPE_RADIUS), orbit_d, stops_d))
    stops = stops_d.get()
    bounded = np.flatnonzero(stops > params.MAX_ITER)
    chosen = int(bounded[0]) if len(bounded) else 0
    k_ref = min(int(stops[chosen]), params.MAX_ITER)
    orbit = orbit_d[chosen]
    rows = max(1, gpu.CHUNK_PIXELS // ny)
    for row0 in range(0, nx, rows):
        nrows = min(rows, nx-row0)
        k(((nrows*ny+gpu.THREADS-1)//gpu.THREADS,), (gpu.THREADS,),
          (xd, wrd, yd, np.int32(ny), np.int32(row0), np.int32(nrows), np.int32(params.MAX_ITER),
           np.float64(params.ESCAPE_RADIUS), out_d, coord_d if coord_d is not None else np.uint64(0),
           orbit, np.int32(k_ref), np.float32(rho), np.int32(bool(cycle_exit)),
           switch_d if switch_d is not None else np.uint64(0)))
    if timing is not None:
        end.record()
        end.synchronize()
        timing["gpu_ms"] = float(cp.cuda.get_elapsed_time(begin, end))
    if diagnostics is not None:
        switches = switch_d.get()
        mask = switches >= 0
        ir, jr = candidates[chosen]
        diagnostics.update(reference_index=[int(ir), int(jr)], reference=[float(x[ir]), float(y[jr])],
                           reference_escaped=bool(stops[chosen] <= params.MAX_ITER), k_ref=k_ref,
                           reference_candidates=len(candidates), rho=rho, reference_radius=radius,
                           fallback_share=float(mask.mean()),
                           initial_fallback_share=float((switches == 0).mean()),
                           mean_switch_iteration=float(switches[mask].mean()) if mask.any() else None)
    out = out_d.get().reshape(nx, ny).view(np.bool_)
    if return_coordinates:
        # Preserve the native representation; do not round ref+gamma to float32.
        return out, {"reference": orbit[0].get(), "gamma": coord_d.get()}
    return out


