"""Run reference copies (the «original computation») and record versions, times and hashes.

usage: python reference/run_reference.py [--jobs N] SET [SET ...]
Each copy runs in its own process: <venv python> -I reference/sets/<SET>/<SET>_ref.py (cwd = that dir).
Writes reference/sets/<SET>/meta.json and run_stderr.txt. Refuses to start a process while
Windows FreeVirtualMemory < 8,388,608 KB (memory rule of this repository's tests).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(HERE))
import make_reference  # noqa: E402
import sets  # noqa: E402

MIN_FREE_KB = 8_388_608
VERSIONS_SNIPPET = (
    "import json,sys,platform,numpy,matplotlib;"
    "from numpy._core import _multiarray_umath as m;"
    "print(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,"
    "'matplotlib':matplotlib.__version__,'numpy_cpu_baseline':list(m.__cpu_baseline__),"
    "'numpy_cpu_dispatch':list(m.__cpu_dispatch__),"
    "'numpy_cpu_features_on':sorted(k for k,v in m.__cpu_features__.items() if v)}))"
)


def child_env():
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(F / ".mplconfig")
    return env


def free_virtual_kb():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def versions():
    out = subprocess.run([sys.executable, "-I", "-c", VERSIONS_SNIPPET], capture_output=True, text=True,
                         check=True, env=child_env()).stdout
    return json.loads(out)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_one(name, vers):
    import numpy as np
    d = make_reference.set_dir(name)
    script = make_reference.script_path(name)
    for old in ("divergence_map.npy", "meta.json"):
        if (d / old).exists():
            raise SystemExit(f"{d / old} exists - move it away before re-running")
    while True:
        free = free_virtual_kb()
        if free >= MIN_FREE_KB:
            break
        print(f"[{name}] FreeVirtualMemory {free} KB < {MIN_FREE_KB}; waiting 60 s", flush=True)
        time.sleep(60)
    t0 = time.perf_counter()
    start = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(d / "run_stderr.txt", "wb") as err:
        rc = subprocess.run([sys.executable, "-I", str(script)], cwd=str(d), stdout=err, stderr=subprocess.STDOUT,
                            env=child_env()).returncode
    wall = time.perf_counter() - t0
    end = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    meta = {"set": name, "params": sets.ALL[name], "script": str(script.relative_to(F)).replace("\\", "/"),
            "script_sha256": sha256_file(script), "returncode": rc, "start": start, "end": end,
            "wall_seconds_whole_script": round(wall, 3), "free_virtual_kb_at_start": free, **vers}
    if rc == 0:
        m = np.load(d / "divergence_map.npy")
        pngs = sorted(p.name for p in d.glob("*.png"))
        meta.update({"map_shape": list(m.shape), "map_dtype": str(m.dtype), "map_true": int(m.sum()),
                     "map_sha256": hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest(),
                     "png": pngs, "png_sha256": {p: sha256_file(d / p) for p in pngs}})
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[{name}] rc={rc} wall={wall:.1f}s {end}", flush=True)
    return name, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    make_reference.generate(a.names)
    vers = versions()
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda n: run_one(n, vers), a.names))
    bad = [n for n, rc in res if rc]
    print("done; failed:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
