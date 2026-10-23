"""One-pixel CPU mirror of the CUDA operation order; never imports CuPy."""
import math
import struct
import numpy as np
from . import cfn, reference

TINY = np.finfo(float).tiny
EPS = np.finfo(float).eps
LOG_R = math.log(1e10)


def mul(a, b):
    return complex(a.real*b.real-a.imag*b.imag, a.real*b.imag+a.imag*b.real)


def divide(a, b):
    if abs(b.real) >= abs(b.imag):
        r = b.imag/b.real
        s = 1/(b.real+b.imag*r)
        return complex((a.real+a.imag*r)*s, (a.imag-a.real*r)*s)
    r = b.real/b.imag
    s = 1/(b.imag+b.real*r)
    return complex((a.real*r+a.imag)*s, (a.imag*r-a.real)*s)


def finite(z):
    return math.isfinite(z.real) and math.isfinite(z.imag)


def ddadd(hi, lo, x):
    s = hi+x
    v = s-hi
    t = ((hi-(s-v))+(x-v))+lo
    h = s+t
    return h, t-(h-s)


def same(a, b):
    return struct.pack('dd', a.real, a.imag) == struct.pack('dd', b.real, b.imag)


def step(ref, e, delta, lam, a, n, escape_rule='margin', max_iter=None):
    """Return (e, delta, escaped, flag, rebased), for the new iteration n."""
    max_iter = len(ref.z)-3 if max_iter is None else max_iter
    j = e+1
    if j >= len(ref.z):
        return e, delta, False, 3, False
    z = complex(ref.z[j])
    w = mul(delta, a)+mul(complex(ref.z[e]), lam)
    direct = (abs(z)<TINY or not finite(z) or not finite(w)
              or w.real>math.log(np.finfo(float).max) or (j==len(ref.z)-1 and (n<max_iter or ref.margin[j]>0.0)))
    u, dn = 0j, 0j
    if not direct:
        u = complex(cfn.cexpm1(w))
        dn = mul(z, u)
        direct = not finite(dn)
    if not direct and n >= 1:
        margin = (float(ref.margin[j])+2.0*(z.real*dn.real+z.imag*dn.imag)
                  +(dn.real*dn.real+dn.imag*dn.imag))
        if math.isfinite(margin):
            escaped = margin>0 if escape_rule=='margin' else abs(z+dn)>1e10
            if escaped:
                return j, dn, True, 0, False
        else:
            direct = True
    cancel = not direct and u.real < -.5
    if direct or cancel:
        h, l = ddadd(float(ref.log_abs[j]), float(ref.log_abs_lo[j]), w.real)
        wmag = abs(delta)*abs(a)+abs(ref.z[e])*abs(lam)
        tol = 1e-12+8.0*EPS*wmag+8.0*EPS*EPS*(abs(ref.log_abs[j])+abs(w.real))
        if not math.isfinite(h) or not finite(w):
            return e, delta, False, 3, False
        if direct and n >= 1:
            gap = (h-LOG_R)+l
            if gap>tol:
                return j, dn, True, 0, False
            if abs(gap)<=tol:
                return e, delta, False, 3, False
        if h+l < math.log(TINY):
            value = 0j
        else:
            value = complex(math.nan, math.nan)
            if cancel and math.log(TINY)<w.real<math.log(np.finfo(float).max):
                value = mul(z, complex(math.exp(w.real)*math.cos(w.imag), math.exp(w.real)*math.sin(w.imag)))
            if not finite(value) or abs(value)<TINY:
                th, tl = ddadd(float(ref.arg[j]), float(ref.arg_lo[j]), w.imag)
                with np.errstate(all='ignore'):
                    r = float(np.exp(h)*np.exp(l))
                ch,sh,cl,sl = math.cos(th),math.sin(th),math.cos(tl),math.sin(tl)
                value = complex(r*(ch*cl-sh*sl),r*(sh*cl+ch*sl))
        if not finite(value):
            return e, delta, False, 3, False
        return 0, value, False, 0, True
    return j, dn, False, 0, False


def pixel(ref, gamma, max_iter, escape_rule='margin', *, cycles=True, diagnostics=False):
    """(diverged, flag, stop); optional statistics/trace for tens of test pixels.

    Only bit-identical e=2 returns repeat the computed trajectory from n=0.
    The hypot escape rule is an intentionally rounded negative control.
    """
    gamma = complex(gamma)
    stats = {'rebases':0, 'cycle_exits':0, 'trace':[]}
    def finish(div, flag, stop):
        value = (div, flag, stop)
        return (value, stats) if diagnostics else value
    if 0 < abs(gamma) < TINY:
        return finish(False, 2, 0)
    lam = complex(cfn.clog1p(divide(gamma, complex(*map(float, ref.center)))))
    a, delta, e, initial = ref.lc+lam, 0j, 0, None
    for n in range(-1, max_iter+1):
        e, delta, escaped, flag, rebased = step(ref, e, delta, lam, a, n, escape_rule, max_iter)
        if diagnostics:
            stats['trace'].append({'n':n, 'e':e, 'delta':delta, 'rebased':rebased})
        stats['rebases'] += int(rebased)
        if flag or escaped:
            return finish(escaped, flag, n if escaped else 0)
        if e==2:
            if initial is None and n==0:
                initial = delta
            elif initial is not None and cycles and same(delta, initial):
                stats['cycle_exits'] = 1
                return finish(False, 0, max_iter)
        if n>=0 and abs(delta)<TINY and abs(lam)<2.**-900 and gamma!=0:
            return finish(False, 2, 0)
    return finish(False, 0, max_iter)


def samples(x0, y0, eps, pixels, max_iter=500, digits=70, ref_pixel=None, escape_rule='margin'):
    ref, _, _ = reference.choose(x0,y0,eps,max_iter,digits,None if ref_pixel is None else [ref_pixel])
    gx, gy = reference.offsets(x0,y0,eps,ref)
    return [pixel(ref,complex(gx[p],gy[q]),max_iter,escape_rule) for p,q in pixels], ref
