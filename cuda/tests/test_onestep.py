"""One-step test: one power c ** z of the kernel's device function vs the NumPy scalar, and the
kernel's |z| vs np.abs (bit-exact)."""
import itertools
import json
import math

import numpy as np
import pytest

import calc
import sets
from mytet_check import V, kinds, np_pow, set_params, step_ok

RADII = (1e3, 1e6, 1e10)


def mkc(re, im):
    """complex array with exactly these parts (re + 1j*im would turn -0 imag into +0 and inf*0 into NaN)."""
    out = np.empty(len(re), dtype=np.complex128)
    out.real, out.imag = re, im
    return out


def special_inputs():
    q = np.array(list(itertools.product(V, repeat=4)), dtype=np.float64)
    # plus bases with a tiny imaginary part whose atan2 lands on/near the smallest subnormal (exact ties included),
    # against exponents where 0 vs 5e-324 in log(a) decides between a number and NaN
    tiny = [k * 5e-324 for k in (1, 2, 3, 5)] + [-5e-324, 1e-310]
    xs = [2.0, 4.0, 3.0, 1.5, 1.9999999999999998, 6.0, 1e10]
    bs = [(0.0, math.inf), (0.0, -math.inf), (1e10, 0.0), (1e300, 0.0), (math.inf, 0.0), (0.5, math.inf)]
    extra = np.array([(x, y, br, bi) for x in xs for y in tiny for br, bi in bs], dtype=np.float64)
    q = np.concatenate([q, extra])
    return mkc(q[:, 0], q[:, 1]), mkc(q[:, 2], q[:, 3])


def coverage(a, b):
    """The required input categories (reference/SETS.md section 4), each must be present."""
    def bits(v):
        return np.asarray(v, dtype=np.float64).view(np.uint64)
    pz, nz = bits(0.0), bits(-0.0)
    zero_a = (a.real == 0) & (a.imag == 0)
    with np.errstate(all="ignore"):
        w = np_pow(a, b)
    cov = {
        "base 0 (+0,+0)": ((bits(a.real) == pz) & (bits(a.imag) == pz)).any(),
        "base 0 (-0,+0)": ((bits(a.real) == nz) & (bits(a.imag) == pz)).any(),
        "base 0 (+0,-0)": ((bits(a.real) == pz) & (bits(a.imag) == nz)).any(),
        "base 0 (-0,-0)": ((bits(a.real) == nz) & (bits(a.imag) == nz)).any(),
        "base 0 with b.re>0 and b.re<0": (zero_a & (b.real > 0)).any() and (zero_a & (b.real < 0)).any(),
        "exponent 0": ((b.real == 0) & (b.imag == 0) & ~zero_a).any(),
        "negative real base imag +0": ((a.real < 0) & (bits(a.imag) == pz)).any(),
        "negative real base imag -0": ((a.real < 0) & (bits(a.imag) == nz)).any(),
        "nan in input": (np.isnan(a.real) | np.isnan(b.imag)).any(),
        "inf in input": (np.isinf(a.imag) | np.isinf(b.real)).any(),
        "overflow to inf from finite input": bool((np.isfinite(a) & np.isfinite(b) & np.isinf(w.real)).any()),
    }
    for n in (-1, 1, 2, 3, 99, 100):
        cov[f"integer exponent {n}"] = ((b.real == n) & (b.imag == 0)).any()
    return cov


def test_special_inputs_cover_required_cases():
    a, b = special_inputs()
    cov = coverage(a, b)
    assert all(cov.values()), {k: v for k, v in cov.items() if not v}


def check(gpu, a, b, run_dir, label):
    w_new = gpu.onestep(a, b)
    ok, w_np = step_ok(a, b, w_new, rule="4")  # SETS.md section 4 judgement
    with np.errstate(all="ignore"):
        verdict_ok = np.ones(len(a), dtype=bool)
        for r in RADII:
            verdict_ok &= (np.abs(w_new) > r) == (np.abs(w_np) > r)
    bad = np.flatnonzero(~(ok & verdict_ok))
    kinds_eq = (kinds(w_new) == kinds(w_np)).all(axis=-1)
    bitwise = (w_new.view(np.uint64).reshape(-1, 2) == w_np.view(np.uint64).reshape(-1, 2)).all(axis=-1)
    summary = {"label": label, "inputs": int(len(a)), "fail": int(len(bad)), "kind_mismatch": int((~kinds_eq).sum()),
               "bitwise_equal": int(bitwise.sum()),
               "examples": [[repr(complex(a[i])), repr(complex(b[i])), repr(complex(w_new[i])), repr(complex(w_np[i]))]
                            for i in bad[:10]]}
    with open(run_dir / "onestep.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    return summary


def test_onestep_special_values(gpu, run_dir):
    a, b = special_inputs()
    s = check(gpu, a, b, run_dir, "V^4")
    assert s["fail"] == 0, s


def trajectory_inputs(name, per_set=40):
    """(c, z_k) pairs from original-operation trajectories (tester calculator) of spread pixels."""
    _, _, p, x, y, _ = set_params(name)
    c = np.asarray(x)[:, np.newaxis] + 1j * np.asarray(y)[np.newaxis, :]
    flat = np.linspace(0, c.size - 1, per_set).astype(np.int64)
    A, B = [], []
    with np.errstate(all="ignore"):
        for f in flat.tolist():
            cv = c.flat[f]
            _, k, traj = calc.pixel(cv, p["max_iter"], p["escape_radius"], record=True)
            A += [cv] * k
            B += traj[:k]
    return np.array(A, dtype=np.complex128), np.array(B, dtype=np.complex128)


@pytest.mark.parametrize("name", sets.MAIN_ORDER)
def test_onestep_trajectory_inputs(gpu, run_dir, name):
    a, b = trajectory_inputs(name)
    s = check(gpu, a, b, run_dir, f"traj:{name}")
    assert s["fail"] == 0, s


def test_abs_bit_exact(gpu):
    rng = np.random.default_rng(12345)
    q = np.array(list(itertools.product(V, repeat=2)), dtype=np.float64)
    z = np.concatenate([mkc(q[:, 0], q[:, 1]),
                        rng.normal(size=200000) * 10.0 ** rng.uniform(-300, 300, 200000)
                        + 1j * rng.normal(size=200000) * 10.0 ** rng.uniform(-300, 300, 200000),
                        rng.uniform(-1e10, 1e10, 200000) + 1j * rng.uniform(-1e10, 1e10, 200000)])
    got = gpu.cabs(z)
    with np.errstate(all="ignore"):
        want = np.array([np.abs(np.complex128(v)) for v in z.tolist()])
    # NaN where NumPy gives NaN; every other value identical in its bits (signed zeros included)
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan)
    assert np.array_equal(got[~nan].view(np.uint64), want[~nan].view(np.uint64))
    assert math.isinf(gpu.cabs(np.array([complex(math.inf, math.nan)]))[0])
