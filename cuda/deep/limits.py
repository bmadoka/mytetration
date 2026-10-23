"""Depth budget measured from the steepest boundary exploration path."""
import math

from .hp import CTX, D, depth_digits, extents

DEEP_BELOW = 1e-13
L = 1e-40
SHALLOW_MAX_ITER = 500
DECADE_ITER = 20
REF_EXTRA_DIGITS = 30


def max_iter_for(eps):
    """20 more powers per decade below 1e-13, rising smoothly with the zoom:
    500 + round(20 * log10(1e-13 / eps)), rounded half up.

    Every power of ten keeps the old per-decade value (1e-15 -> 540, 1e-20 -> 640, 1e-30 -> 840, 1e-40 -> 1040);
    in between the old rule jumped by 20 just below each power of ten, this one adds 0 or 1 per wheel notch
    (eps x 0.9 adds 20 * log10(1 / 0.9) = 0.915), and just below 1e-13 it starts at 500 like the shallow views."""
    depth_digits(eps)  # positive and finite
    if eps >= DEEP_BELOW:
        return SHALLOW_MAX_ITER
    return SHALLOW_MAX_ITER + math.floor(DECADE_ITER * (-math.log10(eps) - 13) + 0.5)


def ref_digits_for(eps):
    return depth_digits(eps) + REF_EXTRA_DIGITS


def view_reason(x0, y0, eps):
    depth_digits(eps)
    if eps < L:
        return 'beyond_L'
    x, y = D(x0), D(y0)
    if CTX.add(CTX.multiply(x, x), CTX.multiply(y, y)) < D('1e-6'):
        return 'near_zero'
    e, ey = extents(eps)
    if CTX.subtract(x, e) < 0 and CTX.subtract(y, ey) <= 0 <= CTX.add(y, ey):
        return 'branch_cut'
    return None
