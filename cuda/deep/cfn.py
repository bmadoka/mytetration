"""Double complex functions. Operation order matches perturb_kernel.cu."""
import numpy as np


def _complex(re, im):
    out = np.empty(np.broadcast_shapes(np.shape(re), np.shape(im)), dtype=np.complex128)
    out.real, out.imag = re, im
    return out[()] if out.ndim == 0 else out


def cexpm1(z):
    z = np.asarray(z, dtype=np.complex128)
    x, y = z.real, z.imag
    with np.errstate(over='ignore', invalid='ignore'):
        s = np.sin(y / 2)
        return _complex(np.expm1(x) * np.cos(y) - 2 * (s * s), np.exp(x) * np.sin(y))


def clog1p(z):
    z = np.asarray(z, dtype=np.complex128)
    u, v = z.real, z.imag
    h = np.hypot(1 + u, v)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        s = 2 * u + (u * u + v * v)
        re = np.where(np.abs(h - 1) <= .5, .5 * np.log1p(s), np.log(h))
    return _complex(re, np.arctan2(v, 1 + u))
