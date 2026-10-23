import datetime as dt
import json
import os
import sys
from pathlib import Path

import pytest

F = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(F / ".mplconfig"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mytet_check as mc  # noqa: E402

if os.environ.get("MYTET_FAULT_REPORT"):  # fault-injection runs: structured per-test record for tests/fault_judge.py
    from fault_judge import pytest_collectreport, pytest_runtest_makereport  # noqa: F401,E402


@pytest.fixture(scope="session")
def run_dir():
    d = F / "bench" / "hybrid_b3" / "test_out" / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    d.mkdir(parents=True, exist_ok=False)
    return d


@pytest.fixture(scope="session")
def gpu(run_dir):
    from mytet_cuda import device
    try:
        g = device.choose()  # MYTET_GPU (nvidia-smi index, default 0)
        free = device.memory(g["uuid"])["free_mib"]  # fresh nvidia-smi query
    except device.GpuChoiceError as e:
        pytest.exit(f"{e}: GPU tests not launched", returncode=3)
    if free < mc.MIN_FREE_MIB:
        pytest.exit(f"GPU {g['index']} ({g['name']}) free memory {free} MiB < {mc.MIN_FREE_MIB} MiB: "
                    "GPU tests not launched", returncode=3)
    import viewer_check
    try:
        viewer_check.require_memory()  # the memory rule also for this parent process's large GPU work
    except viewer_check.MemoryRefused as e:
        pytest.exit(f"{e}: GPU tests not launched", returncode=3)
    from mytet_cuda import gpu as g
    info = g.device_info()
    yield g
    # GPU memory of this process (context + pool), seen as the drop of the GPU's free memory (informational:
    # other processes on that GPU can move it). The CuPy pool itself is capped at gpu.POOL_LIMIT_BYTES.
    import cupy as cp
    free_now, total = cp.cuda.runtime.memGetInfo()
    rec = {"device": info, "free_mib_before_init": free, "free_mib_at_end_with_context": mc.gpu_free_mib(),
           "memGetInfo_free_mib_at_end": free_now // 2**20, "pool_total_mib": cp.get_default_memory_pool().total_bytes() / 2**20,
           "pool_limit_mib": g.POOL_LIMIT_BYTES / 2**20}
    rec["drop_mib"] = rec["free_mib_before_init"] - rec["free_mib_at_end_with_context"]
    (run_dir / "gpu_memory.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


class SetResults:
    """Per-set GPU map + proof, computed once per test session (shared by test_maps and test_e2e)."""

    def __init__(self, gpu, run_dir):
        self.gpu, self.run_dir, self.cache = gpu, run_dir, {}

    def get(self, name):
        if name not in self.cache:
            variant, lits, p, x, y, eps_y = mc.set_params(name)
            ref_map, meta, png = mc.ref(name)
            new_map = self.gpu.compute_map(x, y, p["max_iter"], p["escape_radius"])
            prod_map, prod_iters = self.gpu.compute_map(x, y, p["max_iter"], p["escape_radius"], return_iters=True)
            mism = np.flatnonzero((new_map != ref_map).ravel())
            matched = np.flatnonzero((new_map == ref_map).ravel())
            sample = mc.matched_sample(name, matched)
            pm = mc.prove(name, mism, self.gpu, x, y, p["max_iter"], p["escape_radius"], prod_map, prod_iters)
            ps = mc.prove(name, sample, self.gpu, x, y, p["max_iter"], p["escape_radius"], prod_map, prod_iters)
            r = {"set": name, "variant": variant, "shape": list(ref_map.shape), "pixels": int(ref_map.size),
                 "mismatch": int(len(mism)), "mismatch_ratio": len(mism) / ref_map.size,
                 "mismatch_pixels": mism.tolist()[:1000],
                 "map_eq_iters_run": bool(np.array_equal(new_map, prod_map)),
                 "proof_mismatch": pm, "proof_sample": ps,
                 "new_true": int(new_map.sum()), "ref_true": int(ref_map.sum())}
            self.cache[name] = (r, new_map, ref_map, mism, pm, x, y, p)
            with open(self.run_dir / "set_results.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
        return self.cache[name]


import numpy as np  # noqa: E402


@pytest.fixture(scope="session")
def set_results(gpu, run_dir):
    return SetResults(gpu, run_dir)
