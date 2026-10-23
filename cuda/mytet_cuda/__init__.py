"""CUDA power-tower fractal (static) — same maps/PNGs as DMTPARK/mytetration, computed on the GPU.

Public entry points: `python -m mytet_cuda VARIANT ...` and `divergence_map(...)`.
"""
from . import variants


def divergence_map(variant, x0=None, y0=None, eps=None, n=None, max_iter=None, escape_radius=None):
    """Map for one variant; parameters given as values (None = that variant's default)."""
    from . import gpu
    p = variants.params(variant)
    given = {"x0": x0, "y0": y0, "eps": eps, "n": n, "max_iter": max_iter, "escape_radius": escape_radius}
    p.update({k: v for k, v in given.items() if v is not None})
    x, y, _ = variants.grid(variant, p["x0"], p["y0"], p["eps"], p["n"])
    return gpu.compute_map(x, y, p["max_iter"], p["escape_radius"])
