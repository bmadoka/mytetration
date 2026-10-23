"""Reference device generator («기준 장치», reference/SETS.md).

A reference copy = the original static script with only three kinds of change:
  (1) the top parameter constants (x0, y0, eps, n, max_iter, escape_radius),
  (2) matplotlib backend Agg + removal of plt.show(),
  (3) output path (PNG goes next to the copy; the map is also saved as divergence_map.npy).
The compute function is not touched. `check_copy()` enforces this and the tests call it.

usage: python reference/make_reference.py [SET ...]   (default: every set in sets.ALL)
Writes reference/sets/<SET>/<SET>_ref.py and reference/DIFF.md.
"""
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(HERE))
import sets  # noqa: E402

STATIC_DIR = "Power Tower Fractal (static)"
# Where the original files of DMTPARK/mytetration (commit 21b26204) are looked for, in this order:
# the fork's root (the parent of this cuda/ folder), then upstream/ inside this folder (a private working copy).
ORIGINAL_ROOTS = (F.parent, F / "upstream")


class OriginalNotFound(FileNotFoundError):
    pass


def original_root():
    """The folder that holds the original files (the first of ORIGINAL_ROOTS with a «Power Tower Fractal (static)»
    folder). OriginalNotFound (one line naming both places) when neither has it — never skipped silently."""
    for root in ORIGINAL_ROOTS:
        if (root / STATIC_DIR).is_dir():
            return root
    raise OriginalNotFound(f"original files not found: no '{STATIC_DIR}' folder in "
                           + " or ".join(str(r) for r in ORIGINAL_ROOTS))


PARAM_NAMES = ["x0", "y0", "eps", "n", "max_iter", "escape_radius"]
PARAM_RE = re.compile(r"^(?P<name>x0|y0|eps|n|max_iter|escape_radius) = (?P<val>\S+)(?P<rest>.*)$")
FUNC_RE = re.compile(r"^def compute_tetration_divergence\(.*?^    return divergence_map\n", re.S | re.M)

ADDED_HEADER = ["import os", "import matplotlib", 'matplotlib.use("Agg")']
OUT_DIR_LINE = "_OUT_DIR = os.path.dirname(os.path.abspath(__file__))"
SAVE_LINE = 'np.save(os.path.join(_OUT_DIR, "divergence_map.npy"), divergence_map)'
SAVEFIG_OLD = "plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)"
SAVEFIG_NEW = "plt.savefig(os.path.join(_OUT_DIR, filename), dpi=600, bbox_inches='tight', pad_inches=0)"


def upstream_path(variant):
    return original_root() / STATIC_DIR / f"PTF_static_{variant}.py"


def upstream_text(variant):
    # a Windows checkout with core.autocrlf=true has CRLF; the original bytes (git blob) have LF
    p = upstream_path(variant)
    try:
        return p.read_bytes().decode("utf-8").replace("\r\n", "\n")
    except OSError as e:  # missing or unreadable original file: one line, never skipped
        raise OriginalNotFound(f"cannot read the original file {p} ({type(e).__name__}: {e.strerror or e})") from None


def upstream_defaults(variant):
    """Literal text of the six parameter constants in the original file (first occurrence)."""
    vals = {}
    for line in upstream_text(variant).split("\n"):
        m = PARAM_RE.match(line)
        if m and m["name"] not in vals:
            vals[m["name"]] = m["val"]
    return tuple(vals[k] for k in PARAM_NAMES)


def make_copy(variant, literals):
    src = upstream_text(variant)
    assert "\r" not in src
    want = dict(zip(PARAM_NAMES, literals))
    out, seen = [], set()
    for line in src.split("\n"):
        m = PARAM_RE.match(line)
        if m and m["name"] not in seen:
            seen.add(m["name"])
            line = f"{m['name']} = {want[m['name']]}{m['rest']}"
        if line == "import matplotlib.pyplot as plt":
            out.extend(ADDED_HEADER)
        if line == "plt.show()":
            continue
        if line == SAVEFIG_OLD:
            line = SAVEFIG_NEW
        out.append(line)
        if line.startswith("from matplotlib.colors import LinearSegmentedColormap"):
            out.append(OUT_DIR_LINE)
        if line.startswith("divergence_map = compute_tetration_divergence("):
            out.append(SAVE_LINE)
    assert seen == set(PARAM_NAMES), seen
    return "\n".join(out)


def check_copy(variant, text):
    """Return a list of violations of the «three allowed changes» rule (empty = OK)."""
    src = upstream_text(variant)
    bad = []
    f_up, f_cp = FUNC_RE.search(src), FUNC_RE.search(text)
    if not f_up or not f_cp or f_up.group(0) != f_cp.group(0):
        bad.append("compute_tetration_divergence body differs from upstream")
    allowed_added = set(ADDED_HEADER) | {OUT_DIR_LINE, SAVE_LINE, SAVEFIG_NEW}
    allowed_removed = {"plt.show()", SAVEFIG_OLD}
    for d in difflib.ndiff(src.split("\n"), text.split("\n")):
        tag, line = d[:2], d[2:]
        if tag == "+ ":
            if line in allowed_added or PARAM_RE.match(line):
                continue
            bad.append(f"added line not allowed: {line!r}")
        elif tag == "- ":
            if line in allowed_removed or PARAM_RE.match(line):
                continue
            bad.append(f"removed line not allowed: {line!r}")
    # parameter lines may only change their value, never their trailing comment
    up_params = [PARAM_RE.match(l) for l in src.split("\n") if PARAM_RE.match(l)]
    cp_params = [PARAM_RE.match(l) for l in text.split("\n") if PARAM_RE.match(l)]
    if [(m["name"], m["rest"]) for m in up_params] != [(m["name"], m["rest"]) for m in cp_params]:
        bad.append("parameter line names/comments changed")
    return bad


def set_dir(name):
    return HERE / "sets" / name


def script_path(name):
    return set_dir(name) / f"{name}_ref.py"


def generate(names):
    for name in names:
        variant, *lits = sets.ALL[name]
        text = make_copy(variant, lits)
        bad = check_copy(variant, text)
        assert not bad, (name, bad)
        p = script_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))


def write_diff_md():
    lines = ["# 기준 장치 diff (원본 static 스크립트 → 기준 장치 사본)", "",
             "생성기: `reference/make_reference.py` · 검사: `check_copy()` (tests/test_reference_device.py)",
             "",
             "허용된 변경은 셋뿐이다: ① 맨 위 매개변수 상수 ② matplotlib 백엔드 Agg + `plt.show()` 제거 "
             "③ 출력 경로(PNG를 사본 옆에 쓰고 맵 `divergence_map.npy`도 저장). "
             "`compute_tetration_divergence` 본문은 원본과 바이트까지 같다(검사가 확인).", ""]
    for name in sorted(sets.ALL, key=lambda k: (k[0] != "D", k)):
        p = script_path(name)
        if not p.exists():
            continue
        variant = sets.ALL[name][0]
        up = upstream_text(variant).split("\n")
        cp = p.read_bytes().decode("utf-8").split("\n")
        diff = list(difflib.unified_diff(up, cp, f"Power Tower Fractal (static)/PTF_static_{variant}.py",
                                         f"reference/sets/{name}/{name}_ref.py", lineterm="", n=1))
        lines += [f"## {name} ({variant}) — params {sets.ALL[name][1:]}", "", "```diff", *diff, "```", ""]
    (HERE / "DIFF.md").write_bytes("\n".join(lines).encode("utf-8"))


def main():
    try:
        for v in sets.VARIANTS:
            assert upstream_defaults(v) == sets.DEFAULTS[v], (v, upstream_defaults(v))
        names = sys.argv[1:] or list(sets.ALL)
        generate(names)
        write_diff_md()
    except OriginalNotFound as e:
        sys.exit(f"error: {e}")
    print(f"generated {len(names)} reference copies; DIFF.md written")


if __name__ == "__main__":
    main()
