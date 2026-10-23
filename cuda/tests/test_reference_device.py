"""Reference device: copies differ from the original only in the three allowed ways, and the stored
original-computation outputs come from exactly these copies."""
import hashlib
import json

import numpy as np
import pytest

import make_reference
import sets
from mytet_check import F


@pytest.mark.parametrize("name", sorted(sets.ALL))
def test_copy_is_allowed_diff_and_outputs_match(name):
    variant, *lits = sets.ALL[name]
    path = make_reference.script_path(name)
    text = path.read_bytes().decode("utf-8")
    assert make_reference.check_copy(variant, text) == []
    assert text == make_reference.make_copy(variant, lits)
    meta = json.loads((path.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["returncode"] == 0
    assert meta["script_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    m = np.load(path.parent / "divergence_map.npy")
    assert m.dtype == np.bool_
    assert hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest() == meta["map_sha256"]
    for png, h in meta["png_sha256"].items():
        assert hashlib.sha256((path.parent / png).read_bytes()).hexdigest() == h
    assert len(meta["png"]) == 1
    for k in ("python", "numpy", "matplotlib", "numpy_cpu_features_on"):
        assert meta[k]


def test_check_copy_catches_forbidden_changes():
    """Control: the diff checker must reject a changed compute function and an extra statement."""
    good = make_reference.make_copy("1by1_H", sets.DEFAULTS["1by1_H"])
    assert make_reference.check_copy("1by1_H", good) == []
    bad_func = good.replace("z = c_val ** z", "z = c_val ** z * 1")
    assert make_reference.check_copy("1by1_H", bad_func)
    bad_extra = good.replace("plt.axis('off')", "plt.axis('off')\nprint('x')")
    assert make_reference.check_copy("1by1_H", bad_extra)
    bad_comment = good.replace("eps = 5e1 #x0", "eps = 5e1 #changed")
    assert make_reference.check_copy("1by1_H", bad_comment)


def test_diff_md_covers_every_copy():
    t = (F / "reference" / "DIFF.md").read_text(encoding="utf-8")
    for name in sets.ALL:
        assert f"## {name} (" in t


def test_defaults_are_the_upstream_literals():
    for v in sets.VARIANTS:
        assert make_reference.upstream_defaults(v) == sets.DEFAULTS[v]
