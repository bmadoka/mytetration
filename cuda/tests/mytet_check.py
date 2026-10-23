"""Shared check logic for the tests: sets, one-step comparison, rounding-boundary proof."""
import decimal
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

F = Path(__file__).resolve().parents[1]
for p in (F / "reference", F / "tester", F):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_reference  # noqa: E402
import sets  # noqa: E402
from mytet_cuda import variants  # noqa: E402

TAU0 = 64 * 2.0 ** -53
MIN_FREE_MIB = 2048

# One-step special values (SETS.md section 4).
V = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, 0.5, -0.5, 0.25, 3.7, -3.7, 99.0, 100.0, 101.0, -101.0, 709.0, 710.0,
     1e10, -1e10, 1e20, 1e300, -1e300, 1e-300, 5e-324, math.inf, -math.inf, math.nan]


def gpu_free_mib():
    """Free MiB of the chosen GPU (MYTET_GPU, nvidia-smi index, default 0)."""
    from mytet_cuda import device
    return device.memory(device.choose()["uuid"])["free_mib"]


def set_params(name):
    variant, *lits = sets.SETS[name]
    p = variants.params(variant, **dict(zip(variants.PARAM_NAMES, lits)))
    x, y, eps_y = variants.grid(variant, p["x0"], p["y0"], p["eps"], p["n"])
    return variant, lits, p, x, y, eps_y


def ref(name):
    d = make_reference.set_dir(name)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["returncode"] == 0
    return np.load(d / "divergence_map.npy"), meta, d / meta["png"][0]


def kinds(a):
    """0 finite, 1 inf, 2 nan — per real/imag component."""
    a = np.asarray(a, dtype=np.complex128)
    out = np.zeros(a.shape + (2,), dtype=np.int8)
    for k, part in enumerate((a.real, a.imag)):
        out[..., k] = np.where(np.isnan(part), 2, np.where(np.isinf(part), 1, 0))
    return out


def np_pow(a, b):
    """NumPy complex128 *scalar* power, elementwise (the original's operation)."""
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    with np.errstate(all="ignore"):
        return np.array([np.complex128(u) ** np.complex128(v) for u, v in zip(a.tolist(), b.tolist())],
                        dtype=np.complex128)


def abs_log(a):
    with np.errstate(all="ignore"):
        return np.array([float(np.abs(np.log(np.complex128(u)))) for u in np.asarray(a).ravel().tolist()])


_D = decimal.Context(prec=100, Emax=999999, Emin=-999999)  # far beyond binary64: no overflow / subnormal rounding
_T0 = _D.divide(decimal.Decimal(64), decimal.Decimal(2) ** 53)


def _dec(v):
    return decimal.Decimal(float(v))  # exact conversion of a binary64 value


def tau_rule(b, la):
    """tau of SETS.md section 3 (ii) as an exact-enough Decimal (100 digits), or "nan" / "inf".
    tau = 64*2^-53*(1 + |b|*la), la = |log a| as NumPy gives it; la not finite -> 64*2^-53."""
    br, bi = float(b.real), float(b.imag)
    if not math.isfinite(la):  # |log c| not finite (c = 0, or c NaN) -> tau = 64*2^-53 (SETS.md section 3 (ii))
        return _T0
    # |z_k| by components (never by a float hypot, which overflows for the finite 1.3e308+1.3e308j): an infinite
    # component -> inf (even with a NaN, as np.abs gives); else a NaN component -> NaN; else the exact finite value
    if math.isinf(br) or math.isinf(bi):
        return "nan" if la == 0.0 else "inf"  # inf * 0 is undefined
    if math.isnan(br) or math.isnan(bi):
        return "nan"
    mb = _D.sqrt(_D.add(_D.multiply(_dec(br), _dec(br)), _D.multiply(_dec(bi), _dec(bi))))
    return _D.multiply(_T0, _D.add(1, _D.multiply(mb, _dec(la))))


def _within(d2, w2, tau):
    """sqrt(d2) <= tau*sqrt(w2) with tau Decimal or "inf" (inf * 0 counts as 0)."""
    if tau == "inf":
        return w2 > 0 or d2 == 0
    return _D.sqrt(d2) <= _D.multiply(tau, _D.sqrt(w2))


def _kind(v):
    return 2 if math.isnan(v) else (1 if math.isinf(v) else 0)


def step_one(b, w_new, w_np, la, same_re, same_im, rule="3ii"):
    """The SETS.md section 3 (ii) rule for one step (kinds already equal, not both components bit-identical),
    evaluated in Decimal with 100 digits: no overflow, no subnormal rounding of intermediates."""
    tau = tau_rule(b, la)
    if tau == "nan":
        if rule == "3ii":
            return False  # section 3 (ii): tau NaN -> only bit-identical results pass
        # section 4 (one-step test): kinds per component; a finite component cannot be «within NaN» -> bit-identical
        return all(same or k != 0 for same, k in ((same_re, _kind(w_np.real)), (same_im, _kind(w_np.imag))))
    kp = (_kind(w_np.real), _kind(w_np.imag))
    if kp == (0, 0):
        d2 = _D.add(_D.power(_D.subtract(_dec(w_new.real), _dec(w_np.real)), 2),
                    _D.power(_D.subtract(_dec(w_new.imag), _dec(w_np.imag)), 2))
        w2 = _D.add(_D.power(_dec(w_np.real), 2), _D.power(_dec(w_np.imag), 2))
        return _within(d2, w2, tau)
    for cn, cp, k, same in ((w_new.real, w_np.real, kp[0], same_re), (w_new.imag, w_np.imag, kp[1], same_im)):
        if k != 0 or same:  # non-finite component: equal kind is enough; bit-identical component: passes
            continue
        if not _within(_D.power(_D.subtract(_dec(cn), _dec(cp)), 2), _D.power(_dec(cp), 2), tau):
            return False
    return True


def step_ok(a, b, w_new, w_np=None, rule="3ii"):
    """Per element: SETS.md section 3 (ii) rule (rule="3ii", the proof) or the section 4 one-step test rule
    (rule="4": kinds per component, finite components within tau; with tau NaN the finite components must be
    bit-identical and NaN/inf components need only the same kind). Returns (ok bool[], w_np)."""
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    w_new = np.asarray(w_new, dtype=np.complex128).ravel()
    if w_np is None:
        w_np = np_pow(a, b)
    same_kind = (kinds(w_new) == kinds(w_np)).all(axis=-1)
    same_bits = w_new.view(np.uint64).reshape(-1, 2) == w_np.view(np.uint64).reshape(-1, 2)
    bits_eq = same_bits.all(axis=-1)
    ok = same_kind & bits_eq
    todo = np.flatnonzero(same_kind & ~bits_eq)
    if len(todo):
        la = abs_log(a[todo])
        for k, idx in enumerate(todo.tolist()):
            ok[idx] = step_one(complex(b[idx]), complex(w_new[idx]), complex(w_np[idx]), float(la[k]),
                               bool(same_bits[idx, 0]), bool(same_bits[idx, 1]), rule)
    return ok, w_np


def same_bits_or_nan(u, v):
    """Per component: identical bits, or both NaN (a NaN's sign/payload is not part of its value; NumPy's SIMD
    add picks either operand's NaN pattern depending on the lane)."""
    u, v = np.complex128(u), np.complex128(v)
    for p, q in ((u.real, v.real), (u.imag, v.imag)):
        if np.isnan(p) and np.isnan(q):
            continue
        if np.float64(p).view(np.uint64) != np.float64(q).view(np.uint64):
            return False
    return True


def matched_sample(name, matched_flat):
    m = np.sort(np.asarray(matched_flat, dtype=np.int64))
    step = max(1, len(m) // 200)
    off = zlib.crc32(name.encode("utf-8")) % step
    return m[off::step][:200]


def replay(traj_k, stop, max_iter, escape_radius):
    """Original decision rule applied to z_1..z_stop. Returns (diverged, stop) or None if inconsistent."""
    with np.errstate(all="ignore"):
        esc = np.abs(traj_k[1:stop + 1]) > escape_radius
    hits = np.flatnonzero(esc)
    if len(hits):
        return True, int(hits[0]) + 1
    return False, max_iter if stop == max_iter else -1


def prove(name, pixels, gpu, x, y, max_iter, escape_radius, prod_map, prod_iters):
    """Rounding-boundary proof (SETS.md section 3) for the given flat pixel indices."""
    pixels = np.asarray(pixels, dtype=np.int64)
    ny = len(y)
    div_r, stop_r, traj = gpu.record(x, y, pixels, max_iter, escape_radius)
    c_np = (x[pixels // ny] + (1j * y[np.newaxis, :])[0][pixels % ny])  # original's c, row/col picked
    c_full = variants.full_c(x, y).ravel()[pixels]
    assert c_np.view(np.uint64).tolist() == c_full.view(np.uint64).tolist()
    res = {"pixels": int(len(pixels)), "ok": 0, "fail0": 0, "fail_i": 0, "fail_ii": 0, "fail_iii": 0, "steps": 0,
           "failed_pixels": []}
    for t, p in enumerate(pixels.tolist()):
        K = int(stop_r[t])
        z = traj[t, :K + 1]
        f0 = not same_bits_or_nan(z[0], c_full[t])
        fi = bool(div_r[t]) != bool(prod_map.flat[p]) or K != int(prod_iters.flat[p])
        ok_ii, _ = step_ok(np.full(K, z[0]), z[:K], z[1:K + 1])
        fii = not bool(ok_ii.all())
        rp = replay(traj[t], K, max_iter, escape_radius)
        fiii = rp != (bool(div_r[t]), K)
        res["steps"] += K
        res["fail0"] += f0
        res["fail_i"] += fi
        res["fail_ii"] += fii
        res["fail_iii"] += fiii
        if f0 or fi or fii or fiii:
            if len(res["failed_pixels"]) < 20:
                res["failed_pixels"].append({"pixel": p, "fail0": f0, "fail_i": fi, "fail_ii": fii, "fail_iii": fiii,
                                             "K": K, "first_bad_step": int(np.flatnonzero(~ok_ii)[0]) if fii else None})
        else:
            res["ok"] += 1
    return res
