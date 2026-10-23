"""Run a reference copy's own module-level code with a given map instead of computing it.

Used to make the «기준 PNG» of the end-to-end check (original map with proven mismatch pixels flipped,
drawn by the reference device's picture code) and, as a control, to redraw the unmodified original map.
The only thing removed is the `def compute_tetration_divergence` statement; the call site receives a stub
that checks its arguments against the map shape and returns the given map.
"""
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_with_map(script_path, divergence_map, out_dir):
    """Returns the path of the PNG the reference code wrote into out_dir."""
    script_path = Path(script_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    n_def = sum(isinstance(nd, ast.FunctionDef) and nd.name == "compute_tetration_divergence" for nd in tree.body)
    assert n_def == 1, n_def
    tree.body = [nd for nd in tree.body
                 if not (isinstance(nd, ast.FunctionDef) and nd.name == "compute_tetration_divergence")]
    m = np.array(divergence_map, dtype=bool, copy=True)

    def stub(*args):
        dims = args[:-2]  # (n,) or (nx, ny); then max_iter, escape_radius
        shape = (dims[0], dims[0]) if len(dims) == 1 else tuple(dims)
        assert m.shape == shape, (m.shape, shape)
        return m

    before = set(out_dir.glob("*.png"))
    ns = {"__name__": "__reference_harness__", "__file__": str(out_dir / script_path.name),
          "compute_tetration_divergence": stub}
    plt.close("all")
    try:
        exec(compile(tree, str(script_path), "exec"), ns)
    finally:
        plt.close("all")
    new = sorted(set(out_dir.glob("*.png")) - before)
    assert len(new) == 1, new
    return new[0]
