"""The one-step comparator (mytet_check.step_ok) equals the rule of
reference/SETS.md section 3 (ii) computed with exact rational arithmetic (tests/exact_rule.py) — on three
hand-picked counterexamples, on the special-value grid V (GPU one-step results) and on
deterministic perturbations around tau. Boundary cases (sides within 1e-40) are reported, not used."""
import json
import math

import numpy as np

import exact_rule
from mytet_check import TAU0, abs_log, np_pow, step_ok
from test_onestep import special_inputs


def _nan_with_payload(payload):
    return np.array([payload], dtype=np.uint64).view(np.float64)[0]


def counterexamples():
    a1 = np.complex128(complex(1.0, 5e-324))
    b1 = np.complex128(1.3e308 + 0j)
    with np.errstate(all="ignore"):
        w1 = a1 ** b1
        b2 = np.complex128(complex(2.0 ** 1023, 0.0))
        w2 = a1 ** b2
        a3, b3 = np.complex128(2 + 0j), np.complex128(complex(math.nan, 0.0))
        w3 = a3 ** b3
        a4, b4 = np.complex128(complex(math.nan, 0.0)), np.complex128(1 + 0j)
        w4 = a4 ** b4
        a5, b5, b6 = np.complex128(1j), np.complex128(complex(math.inf, math.nan)), np.complex128(complex(math.nan, math.inf))
        w5, w6 = a5 ** b5, a5 ** b6
        b7 = np.complex128(1.3e308 + 1.3e308j)
    return [
        ("subnormal intermediate (false pass before)", a1, b1, complex(w1.real + 2.0 ** -47, w1.imag + 2.9e-22)),
        ("b = 2^1023 (false reject before)", a1, b2, complex(w2.real, w2.imag + 1.5e-22)),
        ("tau NaN, other NaN payload", a3, b3, complex(_nan_with_payload(0x7FF8000000000001), w3.imag)),
        ("tau NaN, identical bits (control)", a3, b3, complex(w3)),
        ("|log a| NaN -> tau = 64*2^-53: other NaN payload in a non-finite component", a4, b4,
         complex(_nan_with_payload(0x7FF8000000000001), w4.imag)),
        ("|b| = |inf+nan j| = inf -> tau inf: other NaN payload", a5, b5,
         complex(_nan_with_payload(0x7FF8000000000001), w5.imag)),
        ("|b| = |nan+inf j| = inf -> tau inf: other NaN payload", a5, b6,
         complex(_nan_with_payload(0x7FF8000000000001), w6.imag)),
        ("finite b whose float magnitude overflows (1.3e308+1.3e308j): tau stays tiny, -1+0j rejected", a1, b7,
         complex(-1.0, 0.0)),
    ]


# fixed expectations, not taken from either judge: cases 0-1 are numeric (exact rational evaluation;
# case 7 = an earlier regression); cases 2-6 follow from the rule
# text alone (tau finite or inf -> a NaN component needs
# only the same kind; tau NaN -> only bit-identical results pass)
EXPECTED = {0: False, 1: True, 2: False, 3: True, 4: True, 5: True, 6: True, 7: False}


def _compare(a, b, w_new, out, rule="3ii"):
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    w_new = np.asarray(w_new, dtype=np.complex128)
    ok, w_np = step_ok(a, b, w_new, rule=rule)
    la = abs_log(a)
    for i in range(len(a)):
        exp, boundary = exact_rule.judge(complex(b[i]), complex(w_new[i]), complex(w_np[i]), float(la[i]), rule)
        if boundary:
            out["boundary"].append([repr(complex(a[i])), repr(complex(b[i])), repr(complex(w_new[i]))])
            continue
        out["checked"] += 1
        out["true"] += bool(exp)
        if bool(ok[i]) != bool(exp):
            out["disagree"].append([repr(complex(a[i])), repr(complex(b[i])), repr(complex(w_new[i])),
                                    repr(complex(w_np[i])), bool(ok[i]), bool(exp)])


def test_counterexamples_match_exact_rule(run_dir):
    rows = []
    for k, (name, a, b, w) in enumerate(counterexamples()):
        ok, w_np = step_ok([a], [b], [w])
        exp, boundary = exact_rule.judge(complex(b), complex(w), complex(w_np[0]), float(abs_log([a])[0]))
        rows.append({"case": name, "comparator": bool(ok[0]), "exact": bool(exp), "fixed": EXPECTED[k],
                     "boundary": boundary, "w_np": repr(complex(w_np[0]))})
    (run_dir / "comparator_counterexamples.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    assert not any(r["boundary"] for r in rows)
    assert all(r["comparator"] == r["exact"] == r["fixed"] for r in rows), rows


def test_special_grid_and_perturbations_match_exact_rule(gpu, run_dir):
    a, b = special_inputs()
    w_gpu = gpu.onestep(a, b)
    w_np = np_pow(a, b)
    out = {"checked": 0, "true": 0, "boundary": [], "disagree": []}
    bitwise = (w_gpu.view(np.uint64).reshape(-1, 2) == w_np.view(np.uint64).reshape(-1, 2)).all(axis=-1)
    out["gpu_bitwise_equal"] = int(bitwise.sum())
    idx = np.flatnonzero(~bitwise)
    _compare(a[idx], b[idx], w_gpu[idx], out)  # every GPU result that is not bit-identical (section 3 (ii) rule)
    out4 = {"checked": 0, "true": 0, "boundary": [], "disagree": []}
    _compare(a[idx], b[idx], w_gpu[idx], out4, rule="4")  # the same inputs under the section 4 one-step rule
    out["section4"] = out4
    # perturbations around tau on a deterministic subset of finite results, and NaN payload changes
    pa, pb, pw = [], [], []
    la = abs_log(a[::97])
    for k, i in enumerate(range(0, len(a), 97)):
        w = complex(w_np[i])
        if math.isfinite(w.real) and math.isfinite(w.imag) and abs(w) > 0 and math.isfinite(la[k]):
            with np.errstate(all="ignore"):
                tau = TAU0 * (1.0 + abs(complex(b[i])) * la[k])
            if not math.isfinite(tau) or tau * abs(w) == 0:
                continue
            th = (i % 360) * math.pi / 180
            for f in (0.5, 0.999, 1.001, 2.0):
                r = f * tau * abs(w)
                pa.append(a[i]), pb.append(b[i]), pw.append(complex(w.real + r * math.cos(th), w.imag + r * math.sin(th)))
        elif math.isnan(w.real):
            pa.append(a[i]), pb.append(b[i]), pw.append(complex(_nan_with_payload(0x7FF8000000000001), w.imag))
    _compare(pa, pb, pw, out)
    _compare(pa, pb, pw, out4, rule="4")
    out["perturbed_inputs"] = len(pa)
    (run_dir / "comparator_grid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    assert out["checked"] > 1000 and 0 < out["true"] < out["checked"]
    assert out["disagree"] == [], out["disagree"][:5]
    assert out4["checked"] > 1000 and 0 < out4["true"] < out4["checked"] and out4["disagree"] == [], out4["disagree"][:5]
