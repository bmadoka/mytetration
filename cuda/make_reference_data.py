"""Make the reference data the equivalence tests compare against — run once before `pytest`.

1. reference/make_reference.py: a copy of each original static script (Power Tower Fractal (static)/, fork root) per
   comparison set, changed only in the parameter constants, the matplotlib backend and the output path
   (reference/sets/<SET>/<SET>_ref.py, reference/DIFF.md).
2. reference/run_reference.py: runs those copies = the original computation on the CPU (NumPy) -> divergence_map.npy,
   the PNG and meta.json per set (23 sets: 13 comparison sets + 10 small zoom previews).
3. tester/calc.py full: the independent Python re-implementation on the 13 comparison sets -> tester/out/.

Takes about 10-15 minutes with 8 processes (the original computation of the largest set alone is about 5 minutes).
Complete sets are skipped. A set with partial or failed output (e.g. after an interruption) stops this script with
the list of folders/files to delete; nothing is deleted here. No GPU is used.

usage: python make_reference_data.py [--jobs N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

F = Path(__file__).resolve().parent
sys.path[:0] = [str(F / "reference"), str(F / "tester")]
import calc  # noqa: E402
import make_reference  # noqa: E402
import sets  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def reference_state(name):
    """"ok", "missing" or "incomplete" for the original-computation output of one set."""
    d = make_reference.set_dir(name)
    outputs = [d / "meta.json", d / "divergence_map.npy", *d.glob("*.png")]
    if not any(p.exists() for p in outputs):
        return "missing"
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        m = np.load(d / "divergence_map.npy")
        ok = (meta["returncode"] == 0 and meta["script_sha256"] == sha(make_reference.script_path(name).read_bytes())
              and meta["map_sha256"] == sha(np.ascontiguousarray(m).tobytes()) and len(meta["png"]) == 1
              and all(sha((d / p).read_bytes()) == h for p, h in meta["png_sha256"].items()))
    except (OSError, ValueError, KeyError):
        ok = False
    return "ok" if ok else "incomplete"


def tester_state(name):
    npy, js = F / "tester" / "out" / f"{name}_full.npy", F / "tester" / "out" / f"{name}_full.json"
    if not npy.exists() and not js.exists():
        return "missing"
    try:
        info = json.loads(js.read_text(encoding="utf-8"))
        ok = (info["calc_sha256"] == calc.source_sha256() and info["numpy"] == np.__version__
              and info["map_sha256"] == sha(np.load(npy).tobytes()))
    except (OSError, ValueError, KeyError):
        ok = False
    return "ok" if ok else "incomplete"


def states():
    return ({n: reference_state(n) for n in sets.ALL}, {n: tester_state(n) for n in sets.MAIN_ORDER})


def stop_if_incomplete(ref, tst):
    bad = ([str(make_reference.set_dir(n).relative_to(F)) for n, s in ref.items() if s == "incomplete"]
           + [f"tester/out/{n}_full.*" for n, s in tst.items() if s == "incomplete"])
    if bad:
        sys.exit("partial or failed reference output - delete these and run again: " + ", ".join(bad))


def run(args):
    print("+", " ".join(args), flush=True)
    rc = subprocess.run([sys.executable, *args], cwd=str(F)).returncode
    if rc:  # the step printed its own reason (e.g. original files not found); stop here without a traceback
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    a = ap.parse_args()
    t0 = time.time()
    run(["reference/make_reference.py"])
    ref, tst = states()
    stop_if_incomplete(ref, tst)
    todo = [n for n, s in ref.items() if s == "missing"]
    if todo:
        run(["reference/run_reference.py", "--jobs", str(a.jobs), *todo])
    todo = [n for n, s in tst.items() if s == "missing"]
    if todo:
        run(["tester/calc.py", "full", *todo, "--jobs", str(a.jobs)])
    ref, tst = states()
    stop_if_incomplete(ref, tst)
    if any(s != "ok" for s in [*ref.values(), *tst.values()]):
        sys.exit(f"reference data not complete: {ref} {tst}")
    print(f"reference data ready ({len(ref)} reference sets, {len(tst)} tester maps) in {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
