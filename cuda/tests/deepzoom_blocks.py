"""Local block detector (boundary «places», not counts).

Maps are map[px, py] (1920, 1080) bool, px left->right, py bottom->top (the viewer / mytet_cuda layout).
  boundary   = two horizontally or vertically adjacent pixels with different values (one adjacent pair = one boundary).
               Horizontal boundaries H[px, py] (between px and px+1), vertical V[px, py] (between py and py+1).
  recall R   = share of the precise map's boundaries that have a fast-map boundary of the same direction at the same
               place or +-1 pixel along that direction (H: px-1..px+1, same py; V: py-1..py+1, same px).
  precision P= the same with the roles swapped.
  tiles      = 20x20 pixels, 96 x 54 = 5,184; a boundary belongs to the tile of its first pixel.
  target tile= precise or fast map has >= 10 boundaries in the tile.  block tile = target tile with R < 0.6 or P < 0.6
               (R or P of a tile without boundaries on that side counts as 1).
  verdict    = block tiles <= 2 % of target tiles.
"""
import numpy as np

TILE = 20
MIN_BOUNDARIES = 10
MIN_RP = 0.6
MAX_BLOCK_SHARE = 0.02


def boundaries(m):
    m = np.asarray(m, dtype=bool)
    h = np.zeros(m.shape, dtype=bool)
    v = np.zeros(m.shape, dtype=bool)
    h[:-1, :] = m[1:, :] != m[:-1, :]
    v[:, :-1] = m[:, 1:] != m[:, :-1]
    return h, v


def _widen(b, axis):
    """b or b shifted by +-1 along `axis` (no wrap-around)."""
    w = b.copy()
    if axis == 0:
        w[1:, :] |= b[:-1, :]
        w[:-1, :] |= b[1:, :]
    else:
        w[:, 1:] |= b[:, :-1]
        w[:, :-1] |= b[:, 1:]
    return w


def _tiles(a):
    nx, ny = a.shape
    return a.reshape(nx // TILE, TILE, ny // TILE, TILE).sum(axis=(1, 3))


def compare(fast, precise):
    """Dict with per-tile counts and the verdict for one fast map against the precise map of the same parameters."""
    hf, vf = boundaries(fast)
    hp, vp = boundaries(precise)
    n_p = _tiles(hp.astype(np.int32)) + _tiles(vp.astype(np.int32))
    n_f = _tiles(hf.astype(np.int32)) + _tiles(vf.astype(np.int32))
    rec_p = _tiles((hp & _widen(hf, 0)).astype(np.int32)) + _tiles((vp & _widen(vf, 1)).astype(np.int32))
    rec_f = _tiles((hf & _widen(hp, 0)).astype(np.int32)) + _tiles((vf & _widen(vp, 1)).astype(np.int32))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(n_p > 0, rec_p / np.maximum(n_p, 1), 1.0)
        p = np.where(n_f > 0, rec_f / np.maximum(n_f, 1), 1.0)
    target = (n_p >= MIN_BOUNDARIES) | (n_f >= MIN_BOUNDARIES)
    block = target & ((r < MIN_RP) | (p < MIN_RP))
    nt, nb = int(target.sum()), int(block.sum())
    return {"target_tiles": nt, "block_tiles": nb, "block_share": nb / nt if nt else 0.0,
            "ok": bool(nb <= MAX_BLOCK_SHARE * nt), "recall_all": float(rec_p.sum() / max(n_p.sum(), 1)),
            "precision_all": float(rec_f.sum() / max(n_f.sum(), 1))}


def synthetic_block(m, size=10):
    """Nearest-neighbour shrink by `size` and blow-up back (the «blocks» control)."""
    m = np.asarray(m, dtype=bool)
    small = m[::size, ::size]
    return np.repeat(np.repeat(small, size, axis=0), size, axis=1)[:m.shape[0], :m.shape[1]]


def synthetic_shift(m):
    """Shift right by one pixel along x; the first column keeps the precise values (the «1-pixel shift» control)."""
    m = np.asarray(m, dtype=bool)
    out = m.copy()
    out[1:, :] = m[:-1, :]
    return out
