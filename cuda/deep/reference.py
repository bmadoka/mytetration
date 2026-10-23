"""Extended singular orbit 0, 1, C, C**C, ...; CPU preparation only."""
from dataclasses import dataclass
import math
import time

import mpmath as mp
import numpy as np

from . import hp


@dataclass
class Reference:
    pixel: object
    center: tuple
    z: np.ndarray
    log_abs: np.ndarray
    log_abs_lo: np.ndarray
    margin: np.ndarray
    arg: np.ndarray
    arg_lo: np.ndarray
    lc: complex
    n_ref: object
    min_log10_abs: float
    max_log10_abs: float
    seconds: float
    cycle: object = None


def orbit(cx, cy, max_iter, digits, pixel=None, *, copy_cycles=True):
    """e=n+2; only n>=1 escapes. Copy exact mp repeats, never near repeats.

    A private mpmath context keeps concurrent CPU preparation independent.
    """
    start = time.perf_counter()
    ctx = mp.mp.clone()
    ctx.dps = digits
    c = ctx.mpc(str(cx), str(cy))
    lc = ctx.log(c) if c else ctx.mpc(0)
    z = ctx.mpc(0)
    zs, logs, loglo, margins, args, arglo = [], [], [], [], [], []
    seen, cycle, n_ref = {}, None, None
    for e in range(max_iter + 3):
        if copy_cycles and z in seen and not (e>=3 and abs(z)>ctx.mpf('1e10')):
            first = seen[z]
            period = e - first  # // FAULT_1: exact orbit copy period
            cycle = (first, period)
            for j in range(e, max_iter + 3):
                src = first + (j - first) % period
                for values in (zs, logs, loglo, margins, args, arglo):
                    values.append(values[src])
            break
        if e >= 1:
            seen[z] = e
        zs.append(complex(z))
        log = ctx.log(abs(z)) if z else ctx.ninf
        phase = ctx.arg(z) if z else ctx.mpf(0)
        hi, ah = float(log), float(phase)
        logs.append(hi)
        loglo.append(float(log - hi) if ctx.isfinite(log) else 0.)
        args.append(ah)
        arglo.append(float(phase - ah))
        margins.append(float(z.real*z.real + z.imag*z.imag - ctx.mpf('1e20')))
        if e >= 3 and abs(z) > ctx.mpf('1e10'):
            n_ref = e - 2
            break
        if e == 0:
            z = ctx.mpc(1)
        elif e == 1:
            z = c  # c**1 is c, without exp(log(c)) rounding noise.
        else:
            z = (c if z == 1 else ctx.exp(z * lc)) if c else ctx.mpc(0)
    finite_logs = [v for v in logs if math.isfinite(v)]
    return Reference(pixel, (str(cx), str(cy)), np.asarray(zs, np.complex128),
                     np.asarray(logs), np.asarray(loglo), np.asarray(margins),
                     np.asarray(args), np.asarray(arglo), complex(lc), n_ref,
                     min(finite_logs, default=-math.inf)/math.log(10),
                     max(finite_logs, default=-math.inf)/math.log(10),
                     time.perf_counter() - start, cycle)


def range_issue(ref, max_iter):
    """0 at e=0 is deliberate. Finite logs preserve under/overflowed Z."""
    for e in range(1, min(max_iter + 3, len(ref.z))):
        z, logabs = ref.z[e], ref.log_abs[e]
        final_escape = ref.n_ref is not None and e == ref.n_ref + 2
        if (not math.isfinite(logabs) or not math.isfinite(ref.log_abs_lo[e])
                or not math.isfinite(ref.arg[e]) or not math.isfinite(ref.arg_lo[e])
                or math.isnan(z.real) or math.isnan(z.imag)
                or (not final_escape and (math.isinf(z.real) or math.isinf(z.imag)))):
            return {'n': e - 2, 'e': e, 'log10_abs': float(logabs / math.log(10))}
    return None


DEFAULT_PIXELS = tuple(dict.fromkeys([(960, 540)] + [
    (round(i * 1919 / 3), round(j * 1079 / 3)) for i in range(4) for j in range(4)]))


def at_pixel(x0, y0, eps, pixel, max_iter, digits):
    p, q = pixel
    return orbit(hp.grid_x(x0, eps, p), hp.grid_y(y0, eps, q), max_iter, digits, tuple(pixel))


def choose(x0, y0, eps, max_iter, digits, candidates=None, first_survivor=True):
    best, elapsed, tried = None, 0., []
    for pixel in DEFAULT_PIXELS if candidates is None else candidates:
        ref = at_pixel(x0, y0, eps, pixel, max_iter, digits)
        elapsed += ref.seconds
        tried.append({'pixel': list(pixel), 'n_ref': ref.n_ref, 'seconds': ref.seconds})
        score = ref.n_ref if ref.n_ref is not None else max_iter + 1
        oldscore = -1 if best is None else (best.n_ref if best.n_ref is not None else max_iter + 1)
        if score > oldscore:
            best = ref
        if first_survivor and ref.n_ref is None:
            break
    if best is None:
        raise ValueError('no reference candidates')
    return best, elapsed, tried


def offsets(x0, y0, eps, ref):
    """Subtract in the shared 120-digit grid before converting to double."""
    cx, cy = map(hp.D, ref.center)
    gx = [float(hp.CTX.subtract(hp.grid_x(x0, eps, i), cx)) for i in range(1920)]
    gy = [float(hp.CTX.subtract(hp.grid_y(y0, eps, j), cy)) for j in range(1080)]
    return np.asarray(gx), np.asarray(gy)
