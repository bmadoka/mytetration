"""Fixed comparison sets (reference/SETS.md). Single source for the generator, the runner and the tests.

Every value is kept as the *literal text* that is written into the reference copy of the original
script, so the PNG file name (an f-string of x0, y0, eps) is reproduced exactly.
Fields: variant, x0, y0, eps, n, max_iter, escape_radius.
"""

VARIANTS = ["1by1_H", "1by1_R", "16by9_H", "16by9_R", "9by16_H", "9by16_R", "4by5_H", "4by5_R"]

# D1~D8: each original file with its own default parameters (read from the original files by make_reference.py;
# the literals below are asserted to equal the original text).
DEFAULTS = {
    "1by1_H": ("0", "0", "5e1", "500", "500", "1e+10"),
    "1by1_R": ("0", "0", "5e1", "500", "500", "1e+10"),
    "16by9_H": ("0", "0", "5e0", "500", "500", "1e+10"),
    "16by9_R": ("0", "0", "5e0", "500", "500", "1e+10"),
    "9by16_H": ("0", "0", "5e1", "500", "500", "1e+10"),
    "9by16_R": ("0", "0", "5e1", "500", "500", "1e+10"),
    "4by5_H": ("0", "0", "5e1", "500", "500", "1e+10"),
    "4by5_R": ("0", "0", "5e1", "500", "500", "1e+10"),
}

SETS = {f"D{i + 1}": (v, *DEFAULTS[v]) for i, v in enumerate(VARIANTS)}

# P1~P3: small extra sets (other variants, odd sizes, an integer grid).
SETS["P1"] = ("16by9_R", "0.3", "-0.2", "1.5", "64", "37", "1e3")
SETS["P2"] = ("4by5_H", "0", "0", "4", "65", "500", "1e10")
SETS["P3"] = ("1by1_H", "0", "0", "4", "9", "60", "1e6")

# Zoom-window candidates for Z1/Z2 (1by1_H, n=100 preview, max_iter=500, escape_radius=1e+10).
# Fixed BEFORE any preview was computed. Selection rule (also in SETS.md):
#   Z1 = first candidate in this order whose preview boundary ratio >= 15 %;
#   Z2 = next candidate (after Z1, same order) with ratio >= 15 % whose window does not overlap Z1's.
ZOOM_CANDIDATES = {
    "ZC01": ("-0.43138", "0.895038", "0.1"),
    "ZC02": ("-0.43138", "0.895038", "0.01"),
    "ZC03": ("-0.43138", "0.895038", "0.001"),
    "ZC04": ("-0.43137999995", "0.895038", "1e-4"),
    "ZC05": ("1.4", "0.0", "0.1"),
    "ZC06": ("-0.5", "1.5", "0.5"),
    "ZC07": ("0.0", "1.0", "1.0"),
    "ZC08": ("-2.0", "0.0", "1.0"),
    "ZC09": ("0.5", "1.8", "0.25"),
    "ZC10": ("-1.0", "-1.0", "0.5"),
}
PREVIEWS = {k: ("1by1_H", x0, y0, eps, "100", "500", "1e+10") for k, (x0, y0, eps) in ZOOM_CANDIDATES.items()}

# Z1/Z2 are filled in after the previews (by the rule above) and then frozen; see SETS.md.
ZOOM_SELECTED = {"Z1": "ZC01", "Z2": "ZC08"}  # chosen by reference/select_zoom.py (rule above)
for _z, _c in ZOOM_SELECTED.items():
    _x0, _y0, _eps = ZOOM_CANDIDATES[_c]
    SETS[_z] = ("1by1_H", _x0, _y0, _eps, "500", "500", "1e+10")

ALL = {**SETS, **PREVIEWS}

MAIN_ORDER = [f"D{i}" for i in range(1, 9)] + ["Z1", "Z2", "P1", "P2", "P3"]
