"""Command line: python -m mytet_cuda VARIANT [--x0 ...] [--y0 ...] [--eps ...] [--n ...] [--max-iter ...]
[--escape-radius ...] [--out-dir DIR] [--save-map] [--gpu N]

VARIANT is one of 1by1_H 1by1_R 16by9_H 16by9_R 9by16_H 9by16_R 4by5_H 4by5_R. Parameters not given take that
original file's defaults. Values are read as Python number literals, like editing the constants at the top
of the original file (so `--eps 4` names the PNG ..._eps_4.png and `--eps 5e1` names it ..._eps_50.0.png).
--gpu N picks the NVIDIA GPU by its nvidia-smi index (default: environment variable MYTET_GPU, else 0); an index
nvidia-smi does not list is refused before CUDA is started.
"""
import argparse
import os
import sys
import time

from . import variants


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mytet_cuda", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", choices=variants.VARIANTS)
    for name in variants.PARAM_NAMES:
        ap.add_argument("--" + name.replace("_", "-"), dest=name, metavar="LITERAL")
    ap.add_argument("--out-dir", default=".", help="where the PNG goes (default: current directory)")
    ap.add_argument("--save-map", action="store_true", help="also save divergence_map.npy next to the PNG")
    ap.add_argument("--gpu", metavar="N", help="nvidia-smi index of the GPU (default: $MYTET_GPU, else 0)")
    a = ap.parse_args(argv)
    try:
        p = variants.params(a.variant, **{k: getattr(a, k) for k in variants.PARAM_NAMES})
    except ValueError as e:
        ap.error(str(e))

    from . import device
    try:
        device.use(a.gpu) if a.gpu is not None else device.choose()  # before CUDA is loaded
    except device.GpuChoiceError as e:
        print(f"python -m mytet_cuda: error: {e}", file=sys.stderr)
        return 2
    from . import gpu, plot  # import after argument errors so --help needs no GPU

    t0 = time.perf_counter()
    x, y, eps_y = variants.grid(a.variant, p["x0"], p["y0"], p["eps"], p["n"])
    divergence_map = gpu.compute_map(x, y, p["max_iter"], p["escape_radius"])
    t1 = time.perf_counter()
    os.makedirs(a.out_dir, exist_ok=True)
    path = plot.save_png(divergence_map, a.variant, p["x0"], p["y0"], p["eps"], eps_y, a.out_dir)
    if a.save_map:
        import numpy as np
        np.save(os.path.join(a.out_dir, "divergence_map.npy"), divergence_map)
    t2 = time.perf_counter()
    print(f"{path}  map {divergence_map.shape[0]}x{divergence_map.shape[1]}  "
          f"compute {t1 - t0:.3f}s  png {t2 - t1:.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
