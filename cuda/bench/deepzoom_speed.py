"""No-window 1080p method bench, fixed 11 scroll points and unchanged detector.

Default: all methods; --methods pt32 --rhos .05 .01 .2 for a tuning run.
CUDA time includes reference generation/selection for pt32, excludes grid upload,
allocation and map readback. call_ms includes these costs. Diagnostics are untimed.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

F = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(F / "viewer"), str(F / "tests"), str(F)]
import params
import viewer_check as vc
from deepzoom_blocks import compare


def scroll_points(start):
    p = tuple(start)
    points = [(0, p)]
    for i in range(1, 601):
        p = params.apply_wheel(*p, 1 / 6)
        if i % 60 == 0:
            points.append((i // 6, p))
    return points


def memory_status_ex():
    """Real available commit/virtual memory, same >=8 GiB guard; no CIM dependency."""
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_commit",
                "avail_commit", "total_virtual", "avail_virtual", "avail_extended")]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    kb = min(status.avail_commit, status.avail_virtual) // 1024
    if kb < 8388608:
        raise RuntimeError(f"available virtual memory {kb} KB < 8388608 KB")
    return kb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(F / "logs/deepzoom/speed_methods.json"))
    ap.add_argument("--methods", nargs="+", choices=["f32", "f64", "f64+cycle-exit", "pt32"],
                    default=["f32", "f64", "f64+cycle-exit", "pt32"])
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.05])
    ap.add_argument("--reference-radius", type=int, default=4)
    ap.add_argument("--scenes", nargs="+", choices=["D1", "Z1", "DZa", "DZb"])
    ap.add_argument("--notches", nargs="+", type=int)
    ap.add_argument("--memory-status-ex", action="store_true", help="check real Windows commit/virtual memory via API")
    ap.add_argument("--fault-g2", action="store_true", help="in-memory application of the actual g2 payload; no window")
    a = ap.parse_args()
    memory_kb = memory_status_ex() if a.memory_status_ex else vc.require_memory()
    import fast
    if a.fault_g2:
        import fault_injection_viewer as fi
        old, new = fi.FAULTS["g2"][2:4]
        source = (F / "viewer/fast.py").read_text(encoding="utf-8")
        assert source.count(old) == 1
        exec(compile(source.replace(old, new), str(F / "viewer/fast.py"), "exec"), fast.__dict__)
    scenes = json.loads((F / "viewer/scripts/deepzoom_scenes.json").read_text(encoding="utf-8"))["scenes"]
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": {name: hashlib.sha256((F / name).read_bytes()).hexdigest()
                                for name in ("viewer/fast.py", "viewer/fast_kernel.cu", "tests/viewer_check.py",
                                             "bench/deepzoom_pt32.py", "bench/deepzoom_pt32_kernel.cu")},
              "kernel_options": {"f32": list(fast.OPTIONS), "f64": list(fast.F64_OPTIONS),
                                 "pt32": list(fast.OPTIONS)},
              "device": fast.gpu.device_info(), "warmup": 1, "repetitions": 3,
              "shape": [1920, 1080], "cycle_exit_built": True,
              "memory_guard": "GlobalMemoryStatusEx" if a.memory_status_ex else "CIM", "available_virtual_kb": memory_kb,
              "fault": "g2" if a.fault_g2 else None,
              "selection_rule": {"min_relative_spacing": fast.F32_MIN_REL_SPACING,
                                 "min_scale": fast.F32_MIN_SCALE, "max_extent": fast.F32_C_MAX}, "rows": []}
    start = time.perf_counter()
    methods = [(m, m if m != "f64+cycle-exit" else "f64", m in ("f64+cycle-exit", "pt32"), rho)
               for m in a.methods for rho in (a.rhos if m == "pt32" else [None])]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for scene, spec in scenes.items():
        if a.scenes and scene not in a.scenes:
            continue
        for notch, p in scroll_points(spec["start"]):
            if a.notches and notch not in a.notches:
                continue
            pm = fast.precise_map(*p)
            for method, precision, cycle, rho in methods:
                if method == "pt32" and not a.fault_g2:
                    import deepzoom_pt32
                    map_fn = deepzoom_pt32.fast_map
                    kwargs = dict(cycle_exit=cycle, rho=rho, reference_radius=a.reference_radius)
                else:
                    map_fn = fast.fast_map
                    kwargs = dict(precision=precision, cycle_exit=cycle)
                map_fn(*p, **kwargs)
                times, calls = [], []
                for _ in range(3):
                    timing = {}
                    before = time.perf_counter()
                    map_fn(*p, **kwargs, timing=timing)
                    calls.append(1000*(time.perf_counter()-before))
                    times.append(timing["gpu_ms"])
                diag = {} if method == "pt32" and not a.fault_g2 else None
                diagnostic_kwargs = {"diagnostics": diag} if diag is not None else {}
                fm, c = map_fn(*p, **kwargs, return_coordinates=True, **diagnostic_kwargs)
                cc = vc.coordinate_counts(c)
                del c
                row = {"scene": scene, "notches": notch, "params": p, "method": method,
                       "selected_precision": fast.choose_precision(*p),
                       "actual_precision": "pt32" if diag is not None else fast.calls["precision"][-1],
                       "map_sha256": hashlib.sha256(fm.tobytes()).hexdigest(), "gpu_ms": times,
                       "gpu_median_ms": statistics.median(times), "call_ms": calls,
                       "call_median_ms": statistics.median(calls), "coordinates": cc,
                       "blocks": compare(fm, pm), "mismatch": float((fm != pm).mean()),
                       "both_classes": bool(not (pm.any() and (~pm).any()) or (fm.any() and (~fm).any()))}
                if diag is not None:
                    row["perturbation"] = diag
                result["rows"].append(row)
                print(scene, notch, method, rho, round(row["gpu_median_ms"],3),
                      "blocks", row["blocks"]["block_share"], "mismatch", row["mismatch"],
                      "fallback", (diag or {}).get("fallback_share"), flush=True)
            # Retain partial results if interrupted. Never delete artifacts.
            result["wall_s"] = time.perf_counter() - start
            result["pool_bytes"] = fast.pool_bytes()
            out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
