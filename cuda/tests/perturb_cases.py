"""Shared CPU/device function inputs and component-wise high precision oracle."""
import mpmath as mp
import numpy as np

INPUTS = np.asarray([complex(x, y) for a in (1e-45, 1e-40, 1e-30, 1e-20, 1e-15, 1e-8, .1, 1.)
                     for x, y in ((a, a), (-a, a), (a, 0), (0, a), (.5*a*a, a))])
INPUTS = np.concatenate([INPUTS, [-.9+.2j, 2+3j, -.5+.01j]])


def oracle(z, name):
    # At 50 digits avoid forming 1+w or subtracting 1 even inside the oracle.
    with mp.workdps(50):
        x, y = mp.mpf(float(z.real)), mp.mpf(float(z.imag))
        if name == 'cexpm1':
            return complex(mp.expm1(x)*mp.cos(y)-2*mp.sin(y/2)**2, mp.exp(x)*mp.sin(y))
        return complex(mp.log1p(2*x+x*x+y*y)/2, mp.atan2(y, 1+x))


def assert_components(got, z, name):
    expected = oracle(z, name)
    # Cancellation components use an absolute error bound relative to the terms
    # that cancel; all other components (including ~1e-90) use relative error.
    for part in ('real', 'imag'):
        actual, want = getattr(got, part), getattr(expected, part)
        scale = abs(want)
        if part == 'real':
            scale = max(scale, abs(z.real), z.imag*z.imag)
        bound = 4e-15*scale if scale else 1e-300
        assert abs(actual-want) <= bound, f'{name}({z!r}).{part}: {actual!r} != {want!r}; bound={bound}'
