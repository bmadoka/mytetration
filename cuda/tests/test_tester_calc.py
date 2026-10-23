"""Independent tester calculator: tester/calc.py reproduces the reference maps byte-for-byte on all 13 sets,
and its row-subset path equals the same rows of its full computation."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import calc
import sets
from mytet_check import F, ref

STEP = 32


@pytest.mark.parametrize("name", sets.MAIN_ORDER)
def test_tester_full_equals_reference(name):
    """Full tester maps were computed by `tester/calc.py full` (make_reference_data.py runs it)."""
    full = np.load(F / "tester" / "out" / f"{name}_full.npy")
    info = json.loads((F / "tester" / "out" / f"{name}_full.json").read_text(encoding="utf-8"))
    assert info["mode"] == "full"
    # the archived map must come from THIS calculator source, this NumPy and these parameter values
    assert info["calc_sha256"] == calc.source_sha256()
    assert info["numpy"] == np.__version__
    assert info["values"] == [repr(v) for v in calc.parse_set(name)[1]]
    assert info["map_sha256"] == hashlib.sha256(full.tobytes()).hexdigest()
    r, _, _ = ref(name)
    assert full.dtype == r.dtype and full.shape == r.shape
    assert full.tobytes() == r.tobytes()


@pytest.mark.parametrize("name", ["P1", "P2", "P3"])
def test_tester_full_recomputed_small_sets(name):
    r, _, _ = ref(name)
    with np.errstate(all="ignore"):
        assert calc.compute_rows(*calc.parse_set(name)).tobytes() == r.tobytes()


def test_tester_row_subset_equals_full_rows():
    with ProcessPoolExecutor(6) as ex:
        res = dict(ex.map(calc.rows_subset, sets.MAIN_ORDER, [STEP] * len(sets.MAIN_ORDER)))
    for name in sets.MAIN_ORDER:
        full = np.load(F / "tester" / "out" / f"{name}_full.npy")
        assert res[name].tobytes() == full[::STEP].tobytes(), name


def test_tester_comparison_can_fail():
    """Control: changing escape_radius (1e3 -> 1e10) gives a different P1 map, so equality is informative."""
    variant, vals = calc.parse_set("P1")
    vals = list(vals)
    vals[5] = 1e10
    r, _, _ = ref("P1")
    with np.errstate(all="ignore"):
        assert calc.compute_rows(variant, vals).tobytes() != r.tobytes()
