"""Independent exact judge of the one-step rule (reference/SETS.md section 3 (ii)) — regression expectations.

Rational arithmetic only (fractions.Fraction): the square roots in |b|, |w| and |w_new - w| are removed by squaring,
so the decision is exact. `la` is |log a| as NumPy computes it (the rule defines it that way).
A case whose two sides differ by less than 1e-40 (relative, measured in 120-digit Decimal) is a «boundary» case:
it is reported, not used as an expectation.
"""
import decimal
import math
from fractions import Fraction

T0 = Fraction(64, 2 ** 53)
_C = decimal.Context(prec=120, Emax=999999, Emin=-999999)


def _kind(v):
    return 2 if math.isnan(v) else (1 if math.isinf(v) else 0)


def _bits(v):
    import struct
    return struct.pack("<d", v)


def _tau_parts(b, la):
    """("nan"|"inf"|"t0"|"fin", B2, L)."""
    br, bi = b.real, b.imag
    # SETS.md section 3 (ii): «|log c|가 유한하지 않으면(c = 0) tau = 64*2^-53» — any non-finite |log c| (inf or NaN)
    if not math.isfinite(la):
        return "t0", None, None
    # |z_k| = np.abs(z_k) (SETS.md: «|·| = 복소수 크기 np.abs»): inf if any component is inf, NaN if NaN without inf
    if math.isinf(br) or math.isinf(bi):
        return ("nan" if la == 0.0 else "inf"), None, None  # inf * 0 undefined
    if math.isnan(br) or math.isnan(bi):  # |z_k| NaN -> tau NaN («tau가 NaN이면 비트 같음만 통과»)
        return "nan", None, None
    return "fin", Fraction(br) ** 2 + Fraction(bi) ** 2, Fraction(la)


def _within(D2, W2, kind, B2, L):
    """sqrt(D2) <= tau * sqrt(W2), exactly. Returns (decision, boundary)."""
    if kind == "inf":
        return (W2 > 0 or D2 == 0), False
    R = T0 * T0 * W2
    if kind == "t0":
        dec = D2 <= R
        lhs, rhs = _C.sqrt(_dq(D2)), _C.multiply(_dq(T0), _C.sqrt(_dq(W2)))
    else:
        P = D2 - R * (1 + L * L * B2)
        dec = True if P <= 0 else P * P <= 4 * R * R * L * L * B2
        tau = _C.multiply(_dq(T0), _C.add(1, _C.multiply(_C.sqrt(_dq(B2)), _dq(L))))
        lhs, rhs = _C.sqrt(_dq(D2)), _C.multiply(tau, _C.sqrt(_dq(W2)))
    big = max(lhs, rhs)
    boundary = big > 0 and _C.divide(abs(_C.subtract(lhs, rhs)), big) < decimal.Decimal("1e-40")
    return dec, bool(boundary)


def _dq(fr):
    return _C.divide(decimal.Decimal(fr.numerator), decimal.Decimal(fr.denominator))


def judge(b, w_new, w_np, la, rule="3ii"):
    """(decision, boundary) of the rule for one step. b, w_new, w_np: Python complex; la: float."""
    kn = (_kind(w_new.real), _kind(w_new.imag))
    kp = (_kind(w_np.real), _kind(w_np.imag))
    if kn != kp:
        return False, False
    same = (_bits(w_new.real) == _bits(w_np.real), _bits(w_new.imag) == _bits(w_np.imag))
    if all(same):
        return True, False
    kind, B2, L = _tau_parts(b, la)
    if kind == "nan":
        if rule == "3ii":  # section 3 (ii): «tau가 NaN이면 비트 같음만 통과»
            return False, False
        # section 4: «성분마다 종류 같음 · 유한이면 tau 안» — a finite component is never «within NaN» unless identical
        return all(s or k != 0 for s, k in zip(same, kp)), False
    if kp == (0, 0):
        D2 = (Fraction(w_new.real) - Fraction(w_np.real)) ** 2 + (Fraction(w_new.imag) - Fraction(w_np.imag)) ** 2
        W2 = Fraction(w_np.real) ** 2 + Fraction(w_np.imag) ** 2
        return _within(D2, W2, kind, B2, L)
    boundary = False
    for cn, cp, k, s in ((w_new.real, w_np.real, kp[0], same[0]), (w_new.imag, w_np.imag, kp[1], same[1])):
        if k != 0 or s:
            continue
        dec, bnd = _within((Fraction(cn) - Fraction(cp)) ** 2, Fraction(cp) ** 2, kind, B2, L)
        boundary |= bnd
        if not dec:
            return False, boundary
    return True, boundary
