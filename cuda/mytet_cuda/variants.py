"""The eight original static variants: defaults, grid, and parameter-literal parsing.

Grid expressions are copied from the original Power Tower Fractal (static)/PTF_static_*.py so that x, y and
c are bit-identical to the original (checked by tests/test_grid.py against the reference copies).
"""
import ast

import numpy as np

VARIANTS = ("1by1_H", "1by1_R", "16by9_H", "16by9_R", "9by16_H", "9by16_R", "4by5_H", "4by5_R")

# Literal text of the defaults in each original file (x0, y0, eps, n, max_iter, escape_radius).
DEFAULT_LITERALS = {
    "1by1_H": ("0", "0", "5e1", "500", "500", "1e+10"),
    "1by1_R": ("0", "0", "5e1", "500", "500", "1e+10"),
    "16by9_H": ("0", "0", "5e0", "500", "500", "1e+10"),
    "16by9_R": ("0", "0", "5e0", "500", "500", "1e+10"),
    "9by16_H": ("0", "0", "5e1", "500", "500", "1e+10"),
    "9by16_R": ("0", "0", "5e1", "500", "500", "1e+10"),
    "4by5_H": ("0", "0", "5e1", "500", "500", "1e+10"),
    "4by5_R": ("0", "0", "5e1", "500", "500", "1e+10"),
}

# eps_y = eps * (a/b) and ny = int(n*(a/b)) in the original files.
_RATIO = {"16by9_H": (9, 16), "16by9_R": (16, 9), "9by16_H": (16, 9), "9by16_R": (9, 16),
          "4by5_H": (5, 4), "4by5_R": (4, 5)}

PARAM_NAMES = ("x0", "y0", "eps", "n", "max_iter", "escape_radius")


def parse_literal(text, name):
    """Parse a parameter the way the original's `name = <literal>` line would (int stays int)."""
    try:
        v = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        raise ValueError(f"{name}: not a number literal: {text!r}") from None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name}: not an int/float literal: {text!r}")
    if name in ("n", "max_iter") and not isinstance(v, int):
        raise ValueError(f"{name}: must be an integer literal: {text!r}")
    return v


def params(variant, **literals):
    """Defaults of `variant`, overridden by the given literal strings. Returns a dict of values."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {', '.join(VARIANTS)}")
    out = {}
    for name, default in zip(PARAM_NAMES, DEFAULT_LITERALS[variant]):
        text = literals.get(name)
        out[name] = parse_literal(default if text is None else text, name)
    return out


def grid(variant, x0, y0, eps, n):
    """Returns (x, y, eps_y): x has nx points, y has ny points, eps_y is None for 1by1."""
    if variant in ("1by1_H", "1by1_R"):
        x = np.linspace(x0 - eps, x0 + eps, n)
        y = np.linspace(y0 - eps, y0 + eps, n)
        return x, y, None
    a, b = _RATIO[variant]
    eps_y = eps * (a / b)
    nx, ny = n, int(n * (a / b))
    x = np.linspace(x0 - eps, x0 + eps, nx)
    y = np.linspace(y0 - eps_y, y0 + eps_y, ny)
    return x, y, eps_y


def full_c(x, y):
    """c exactly as the original builds it."""
    return x[:, np.newaxis] + 1j * y[np.newaxis, :]
