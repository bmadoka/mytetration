"""Tester calculator («시험용 계산기», reference/SETS.md section 5): a separate Python re-implementation of the
original computation that can also (a) compute only some x-rows and (b) return trajectories.
It uses only the original's operations: np.complex128 scalar `c_val ** z` and `np.abs(z) > escape_radius`.
Before use it must reproduce the reference maps byte-for-byte (tests/test_tester_calc.py).

usage:
  python tester/calc.py full SET [SET ...] [--jobs N]     -> tester/out/<SET>_full.npy (+ timing json)
  python tester/calc.py rows SET STEP [--jobs N]           -> tester/out/<SET>_rows<STEP>.npy (rows 0, STEP, 2*STEP, ...)
SET is a name from reference/sets.py, or "VARIANT:x0:y0:eps:n:max_iter:escape_radius" (literals).
"""
import argparse
import ast
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F = HERE.parent
OUT = HERE / "out"

_RATIO = {"16by9_H": (9, 16), "16by9_R": (16, 9), "9by16_H": (16, 9), "9by16_R": (9, 16),
          "4by5_H": (5, 4), "4by5_R": (4, 5)}


def source_sha256():
    """sha256 of this file: archived outputs are only valid for the calculator that produced them."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_set(spec):
    if ":" in spec:
        variant, *lits = spec.split(":")
    else:
        sys.path.insert(0, str(F / "reference"))
        import sets
        variant, *lits = sets.ALL[spec]
    vals = [ast.literal_eval(t) for t in lits]
    return variant, vals


def grid(variant, x0, y0, eps, n):
    """x, y exactly as in the original files."""
    if variant in ("1by1_H", "1by1_R"):
        return np.linspace(x0 - eps, x0 + eps, n), np.linspace(y0 - eps, y0 + eps, n)
    a, b = _RATIO[variant]
    eps_y = eps * (a / b)
    nx, ny = n, int(n * (a / b))
    return np.linspace(x0 - eps, x0 + eps, nx), np.linspace(y0 - eps_y, y0 + eps_y, ny)


def pixel(c_val, max_iter, escape_radius, record=False):
    """(diverged, stop number, trajectory z_0..z_stop or None) — the original inner loop."""
    z = c_val
    traj = [z] if record else None
    for k in range(max_iter):
        z = c_val ** z
        if record:
            traj.append(z)
        if np.abs(z) > escape_radius:
            return True, k + 1, traj
    return False, max_iter, traj


def compute_rows(variant, vals, rows=None):
    """Map rows `rows` (default all) as an array (len(rows), ny)."""
    x0, y0, eps, n, max_iter, escape_radius = vals
    x, y = grid(variant, x0, y0, eps, n)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    rows = range(len(x)) if rows is None else list(rows)
    out = np.zeros((len(rows), len(y)), dtype=bool)
    for r, i in enumerate(rows):
        for j in range(len(y)):
            out[r, j] = pixel(c[i, j], max_iter, escape_radius)[0]
    return out


def rows_subset(spec, step):
    """(spec, map of x-rows 0, step, 2*step, ...) — the row-subset path."""
    variant, vals = parse_set(spec)
    nx = len(grid(variant, *vals[:4])[0])
    with np.errstate(all="ignore"):
        return spec, compute_rows(variant, vals, range(0, nx, step))


def _job(args):
    mode, spec, step = args
    variant, vals = parse_set(spec)
    t0 = time.perf_counter()
    if mode == "full":
        m = compute_rows(variant, vals)
        name = f"{spec}_full"
    else:
        nx = len(grid(variant, *vals[:4])[0])
        m = compute_rows(variant, vals, range(0, nx, step))
        name = f"{spec}_rows{step}"
    wall = time.perf_counter() - t0
    safe = name.replace(":", "_")
    OUT.mkdir(exist_ok=True)
    np.save(OUT / f"{safe}.npy", m)
    (OUT / f"{safe}.json").write_text(json.dumps({"spec": spec, "mode": mode, "step": step, "shape": list(m.shape),
                                                  "seconds": round(wall, 3), "values": [repr(v) for v in vals],
                                                  "calc_sha256": source_sha256(), "numpy": np.__version__,
                                                  "map_sha256": hashlib.sha256(m.tobytes()).hexdigest()}),
                                      encoding="utf-8")
    return name, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "rows"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "full":
        jobs = [("full", s, 0) for s in a.args]
    else:
        jobs = [("rows", a.args[0], int(a.args[1]))]
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, wall in ex.map(_job, jobs):
            print(f"{name}: {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
