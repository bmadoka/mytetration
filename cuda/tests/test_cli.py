"""Same usage as the originals: every parameter of the 8 originals is accepted, a variant alone gives that file's
defaults, and the PNG name follows the original f-string."""
import subprocess
import sys

import pytest

import make_reference
from mytet_check import F
from mytet_cuda import plot, variants


@pytest.mark.parametrize("variant", variants.VARIANTS)
def test_defaults_match_upstream_file(variant):
    assert variants.DEFAULT_LITERALS[variant] == make_reference.upstream_defaults(variant)
    p = variants.params(variant)
    ns = {}
    up = make_reference.upstream_text(variant)
    for line in up.split("\n"):
        if make_reference.PARAM_RE.match(line):
            exec(line, ns)
    assert p == {k: ns[k] for k in variants.PARAM_NAMES}
    assert all(type(p[k]) is type(ns[k]) for k in variants.PARAM_NAMES)


def test_literal_semantics_and_png_name():
    p = variants.params("4by5_H", eps="4", x0="-0.2", escape_radius="1e10")
    assert p["eps"] == 4 and type(p["eps"]) is int and p["x0"] == -0.2 and p["escape_radius"] == 1e10
    assert plot.png_name("4by5_H", p["x0"], p["y0"], p["eps"]) == "mytetration_x_-0.2_y_0_eps_4.png"
    d = variants.params("1by1_R")
    assert plot.png_name("1by1_R", d["x0"], d["y0"], d["eps"]) == "mytetration_x_0_y_0_eps_50.0_rotated.png"
    for bad in [dict(eps="abc"), dict(n="1.5"), dict(max_iter="True"), dict(x0="[1]")]:
        with pytest.raises(ValueError):
            variants.params("1by1_H", **bad)
    with pytest.raises(ValueError):
        variants.params("2by3_H")


def test_cli_help_and_rejects_bad_input():
    h = subprocess.run([sys.executable, "-m", "mytet_cuda", "--help"], cwd=str(F), capture_output=True, text=True)
    assert h.returncode == 0
    for opt in ("--x0", "--y0", "--eps", "--n", "--max-iter", "--escape-radius", "--out-dir"):
        assert opt in h.stdout
    for v in variants.VARIANTS:
        assert v in h.stdout
    b = subprocess.run([sys.executable, "-m", "mytet_cuda", "1by1_H", "--n", "1.5"], cwd=str(F),
                       capture_output=True, text=True)
    assert b.returncode == 2 and "n: must be an integer literal" in b.stderr
