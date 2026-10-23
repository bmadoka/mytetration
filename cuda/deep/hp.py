"""Decimal view state and the single definition of the deep pixel grid."""
import math
from decimal import Context, Decimal, ROUND_HALF_EVEN

from viewer import params

HP_DIGITS = 120
CTX = Context(prec=HP_DIGITS, rounding=ROUND_HALF_EVEN)


def D(v):
    return v if isinstance(v, Decimal) else Decimal(v)


def from_floats(x0, y0, eps):
    return D(x0), D(y0), float(eps)


def from_strings(x0s, y0s, eps):
    return D(x0s), D(y0s), float(eps)


def extents(eps):
    e = D(eps)
    return e, CTX.divide(CTX.multiply(e, D(9)), D(16))


def spacing(eps):
    e, ey = extents(eps)
    return CTX.divide(CTX.multiply(D(2), e), D(1919)), CTX.divide(CTX.multiply(D(2), ey), D(1079))


def _grid(center, extent, index, last):
    center = D(center)
    # Pin both endpoints, as linspace does; avoid accumulated division rounding.
    if index == last:
        return CTX.add(center, extent)
    step = CTX.divide(CTX.multiply(D(2), extent), D(last))
    return CTX.add(CTX.subtract(center, extent), CTX.multiply(D(index), step))


def grid_x(x0, eps, px):
    return _grid(x0, extents(eps)[0], px, 1919)


def grid_y(y0, eps, py):
    return _grid(y0, extents(eps)[1], py, 1079)


def apply_drag(hp, dx, dy):
    x0, y0, eps = hp
    sx, sy = spacing(eps)
    return (CTX.subtract(x0, CTX.multiply(D(dx), sx)),
            CTX.add(y0, CTX.multiply(D(dy), sy)), eps)


def apply_wheel(hp, steps):
    return params.apply_wheel(*hp, steps)


def apply_event(hp, ev):
    if ev['type'] == 'drag':
        return apply_drag(hp, ev['dx'], ev['dy'])
    if ev['type'] == 'wheel':
        return apply_wheel(hp, ev['steps'])
    raise ValueError(ev['type'])


def to_record(hp):
    return [str(hp[0]), str(hp[1]), float(hp[2]), float(hp[2]).hex()]


def depth_digits(eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be positive and finite')
    return max(0, math.ceil(-math.log10(eps)))
