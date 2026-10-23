"""Picture path: the original map of each of the 8 variants, drawn by the new picture path,
decodes to the same pixels as the original PNG (size, orientation, rotation, colour)."""
import numpy as np
import pytest
from PIL import Image

import make_reference
import plot_harness
from mytet_check import ref, set_params
from mytet_cuda import plot

D_SETS = [f"D{i}" for i in range(1, 9)]


def pixels(path):
    im = Image.open(path)
    return im.mode, im.size, np.asarray(im)


@pytest.mark.parametrize("name", D_SETS)
def test_new_picture_path_equals_original_png(run_dir, name):
    variant, _, p, x, y, eps_y = set_params(name)
    ref_map, meta, ref_png = ref(name)
    out = run_dir / "plot_path" / name
    out.mkdir(parents=True)
    path = plot.save_png(ref_map, variant, p["x0"], p["y0"], p["eps"], eps_y, str(out))
    assert path.endswith(ref_png.name)
    m1, s1, a1 = pixels(path)
    m0, s0, a0 = pixels(ref_png)
    assert (m1, s1) == (m0, s0)
    assert np.array_equal(a1, a0)


@pytest.mark.parametrize("name", ["D1", "D4"])
def test_picture_check_is_sensitive(run_dir, name):
    """Controls: harness redraw of the original map is identical; one flipped map pixel or a transposed
    map gives a different PNG."""
    variant, _, p, x, y, eps_y = set_params(name)
    ref_map, meta, ref_png = ref(name)
    a0 = pixels(ref_png)[2]
    h = plot_harness.render_with_map(make_reference.script_path(name), ref_map, run_dir / "harness_ctrl" / name)
    assert np.array_equal(pixels(h)[2], a0)
    flipped = ref_map.copy()
    flipped[ref_map.shape[0] // 2, ref_map.shape[1] // 3] ^= True
    out = run_dir / "plot_ctrl" / name
    out.mkdir(parents=True)
    (out / "flip").mkdir()
    p1 = plot.save_png(flipped, variant, p["x0"], p["y0"], p["eps"], eps_y, str(out / "flip"))
    assert not np.array_equal(pixels(p1)[2], a0)
    (out / "rot").mkdir()
    rotated = np.ascontiguousarray(np.rot90(ref_map, 2))
    p2 = plot.save_png(rotated, variant, p["x0"], p["y0"], p["eps"], eps_y, str(out / "rot"))
    assert not np.array_equal(pixels(p2)[2], a0)
