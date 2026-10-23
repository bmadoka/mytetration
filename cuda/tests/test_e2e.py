"""End-to-end: D1~D8 and P1~P3 through the real command line; the PNG decodes to the same
pixels as the reference PNG = original map with the proven mismatch pixels flipped, drawn by the reference
device's own picture code (reference/plot_harness.py)."""
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import make_reference
import plot_harness
import sets
from mytet_check import F, ref

E2E_SETS = [f"D{i}" for i in range(1, 9)] + ["P1", "P2", "P3"]
ARGS = ("--x0", "--y0", "--eps", "--n", "--max-iter", "--escape-radius")


def cli_args(name):
    variant, *lits = sets.SETS[name]
    if name.startswith("D"):
        return [variant]  # defaults path: the variant alone must give the original file's defaults
    return [variant] + [a for opt, lit in zip(ARGS, lits) for a in (f"{opt}={lit}",)]


@pytest.mark.parametrize("name", E2E_SETS)
def test_end_to_end(set_results, run_dir, name):
    r, new_map, ref_map, mism, pm, *_ = set_results.get(name)
    assert pm["ok"] == pm["pixels"] == len(mism)
    expected_map = ref_map.copy()
    expected_map.flat[mism] ^= True  # all mismatches are proven (asserted above)
    exp_png = plot_harness.render_with_map(make_reference.script_path(name), expected_map,
                                           run_dir / "e2e_expected" / name)
    if len(mism) == 0:
        assert np.array_equal(np.asarray(Image.open(exp_png)), np.asarray(Image.open(ref(name)[2])))
    out = run_dir / "e2e" / name
    out.mkdir(parents=True)
    cmd = [sys.executable, "-m", "mytet_cuda", *cli_args(name), "--out-dir", str(out)]
    cp = subprocess.run(cmd, cwd=str(F), capture_output=True, text=True, timeout=600)
    assert cp.returncode == 0, cp.stderr
    pngs = sorted(out.glob("*.png"))
    assert [p.name for p in pngs] == [ref(name)[2].name]
    a, b = Image.open(pngs[0]), Image.open(exp_png)
    assert (a.mode, a.size) == (b.mode, b.size)
    assert np.array_equal(np.asarray(a), np.asarray(b))
