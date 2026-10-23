"""Map judgement: per set, new map vs original map; every mismatch pixel and 200 matched pixels pass the
rounding-boundary proof (0)-(iii) of SETS.md section 3."""
import math

import numpy as np
import pytest

import calc
import sets
from mytet_check import prove, step_ok


@pytest.mark.parametrize("name", sets.MAIN_ORDER)
def test_map_and_proof(set_results, name):
    r, new_map, ref_map, mism, pm, *_ = set_results.get(name)
    assert new_map.shape == ref_map.shape and new_map.dtype == np.bool_
    assert r["map_eq_iters_run"], "product map with iters output differs from plain product map"
    assert pm["pixels"] == r["mismatch"]
    assert pm["ok"] == pm["pixels"], pm  # every mismatch pixel proven
    ps = r["proof_sample"]
    assert ps["pixels"] == min(200, ref_map.size - r["mismatch"])
    assert ps["ok"] == ps["pixels"], ps


SPECIAL = np.array([math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 1e300, -1e300, 5e-324])


@pytest.mark.parametrize("max_iter,radius", [(50, 1e10), (7, 1e3)])
def test_special_grid_nan_inf(gpu, max_iter, radius):
    """NaN/inf/0 coordinates: kernel verdicts and stop numbers equal the original loop's, all pixels proven."""
    x, y = SPECIAL, SPECIAL
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    want = np.zeros(c.shape, dtype=bool)
    want_k = np.zeros(c.shape, dtype=np.int32)
    with np.errstate(all="ignore"):
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                d, k, _ = calc.pixel(c[i, j], max_iter, radius)
                want[i, j], want_k[i, j] = d, k
    got, got_k = gpu.compute_map(x, y, max_iter, radius, return_iters=True)
    assert np.array_equal(got, want)
    assert np.array_equal(got_k, want_k)
    assert np.array_equal(gpu.compute_map(x, y, max_iter, radius), got)
    res = prove("special", np.arange(c.size), gpu, x, y, max_iter, radius, got, got_k)
    assert res["ok"] == c.size, res
    # the grid must actually exercise NaN trajectories that never escape
    with np.errstate(all="ignore"):
        nan_pixels = [(i, j) for i in range(c.shape[0]) for j in range(c.shape[1])
                      if not want[i, j] and np.isnan(calc.pixel(c[i, j], max_iter, radius, record=True)[2][-1])]
    assert nan_pixels


def test_proof_detects_wrong_trajectory():
    """Control for (ii): a trajectory step off by more than tau, or of another kind, is rejected."""
    c = np.complex128(-0.43 + 0.89j)
    z = np.complex128(0.3 - 0.1j)
    w = c ** z
    ok, _ = step_ok([c, c, c], [z, z, z], [w, w * (1 + 1e-12), complex(math.nan, w.imag)])
    assert ok.tolist() == [True, False, False]
    # mixed finite/inf result: the finite component is still checked (2 ** 1e10 = inf + 0j)
    two, big = np.complex128(2 + 0j), np.complex128(1e10 + 0j)
    ok, w_np = step_ok([two, two, two], [big, big, big],
                       [complex(math.inf, 0.0), complex(math.inf, 12345.0), complex(math.inf, math.nan)])
    assert complex(w_np[0]) == complex(math.inf, 0.0)
    assert ok.tolist() == [True, False, False]
    # huge finite results: the comparison must not overflow into inf <= inf
    a, one = np.complex128(1.3e308 + 1.3e308j), np.complex128(1 + 0j)
    ok, w_np = step_ok([a, a, a], [one, one, one],
                       [a, complex(-1.3e308, 1.3e308), complex(1.3e308 * (1 + 2 ** -50), 1.3e308)])
    assert complex(w_np[0]) == complex(a)
    assert ok.tolist() == [True, False, True]
    # |b| itself overflows but tau stays tiny: a gross error must still fail
    a, b = np.complex128(complex(1.0, 5e-324)), np.complex128(1.3e308 + 1.3e308j)
    with np.errstate(all="ignore"):
        exact = a ** b
    ok, _ = step_ok([a, a], [b, b], [exact, complex(-1.0, 0.0)])
    assert ok.tolist() == [True, False]


def test_multi_chunk_launch_equals_single(gpu, monkeypatch):
    """Maps that need several launches (4K) must equal the one-launch map: force tiny chunks on Z1 and P2."""
    import mytet_check as mc
    for name in ("Z1", "P2"):
        _, _, p, x, y, _ = mc.set_params(name)
        one, it1 = gpu.compute_map(x, y, p["max_iter"], p["escape_radius"], return_iters=True)
        monkeypatch.setattr(gpu, "CHUNK_PIXELS", 3 * len(y) + 7)  # 3 rows per launch, last launch partial
        many, it2 = gpu.compute_map(x, y, p["max_iter"], p["escape_radius"], return_iters=True)
        monkeypatch.undo()
        assert np.array_equal(one, many) and np.array_equal(it1, it2), name
