"""Z1/Z2 selection from the n=100 reference previews (rule fixed in sets.py before the previews ran).

boundary ratio = fraction of pixels having at least one 4-neighbour (inside the grid) with a different
verdict. Prints a table and the selection; exit 1 if fewer than two windows qualify.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sets  # noqa: E402

THRESHOLD = 0.15


def boundary_ratio(m):
    m = np.asarray(m, dtype=bool)
    b = np.zeros_like(m)
    b[1:, :] |= m[1:, :] != m[:-1, :]
    b[:-1, :] |= m[:-1, :] != m[1:, :]
    b[:, 1:] |= m[:, 1:] != m[:, :-1]
    b[:, :-1] |= m[:, :-1] != m[:, 1:]
    return float(b.mean())


def window(name):
    _, x0, y0, eps = sets.PREVIEWS[name][:4]
    x0, y0, eps = float(x0), float(y0), float(eps)
    return x0 - eps, x0 + eps, y0 - eps, y0 + eps


def overlap(a, b):
    wa, wb = window(a), window(b)
    return wa[0] <= wb[1] and wb[0] <= wa[1] and wa[2] <= wb[3] and wb[2] <= wa[3]


def main():
    rows, ok = [], []
    for name in sets.ZOOM_CANDIDATES:
        d = HERE / "sets" / name
        m = np.load(d / "divergence_map.npy")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        r = boundary_ratio(m)
        rows.append((name, sets.ZOOM_CANDIDATES[name], m.shape, float(m.mean()), r, meta["map_sha256"]))
        if r >= THRESHOLD:
            ok.append(name)
    z1 = ok[0] if ok else None
    z2 = next((c for c in ok[ok.index(z1) + 1:] if not overlap(z1, c)), None) if z1 else None
    for name, (x0, y0, eps), shape, frac, r, h in rows:
        print(f"| {name} | {x0} | {y0} | {eps} | {shape[0]}x{shape[1]} | {frac:.4f} | {r:.4f} | "
              f"{'yes' if r >= THRESHOLD else 'no'} | {h[:16]} |")
    print(f"Z1 = {z1}  Z2 = {z2}")
    sys.exit(0 if z1 and z2 else 1)


if __name__ == "__main__":
    main()
