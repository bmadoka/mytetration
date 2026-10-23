"""Grid: the new implementation's x, y and c are bit-identical to what each reference copy builds."""
import ast
import os

import numpy as np
import pytest

import make_reference
import sets
from mytet_check import set_params
from mytet_cuda import variants


def reference_c(name):
    """Execute the reference copy's own top-level constants and the first three statements of its
    compute function (x = ..., y = ..., c = ...) and return its c."""
    path = make_reference.script_path(name)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ns = {"np": np, "os": os, "__file__": str(path)}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_tetration_divergence":
            first = node.body[:3]
            assert [t.targets[0].id for t in first] == ["x", "y", "c"]
            exec(compile(ast.Module(body=first, type_ignores=[]), "<grid>", "exec"), ns)
            return ns["c"]
        if isinstance(node, ast.Assign):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<const>", "exec"), ns)
    raise AssertionError("no compute function")


def bits(a):
    return np.ascontiguousarray(a, dtype=np.complex128).view(np.uint64)


@pytest.mark.parametrize("name", sets.MAIN_ORDER)
def test_host_grid_bit_identical(name):
    _, _, _, x, y, _ = set_params(name)
    c_ref = reference_c(name)
    c = variants.full_c(x, y)
    assert c.shape == c_ref.shape == np.load(make_reference.set_dir(name) / "divergence_map.npy").shape
    assert np.array_equal(bits(c), bits(c_ref))


@pytest.mark.parametrize("name", sets.MAIN_ORDER)
def test_gpu_grid_bit_identical(gpu, name):
    _, _, _, x, y, _ = set_params(name)
    assert np.array_equal(bits(gpu.grid_c(x, y)), bits(reference_c(name)))
