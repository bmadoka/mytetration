"""No-window exact event-state audit (full sweep on request).

Local cheap validation: python -B bench/deepzoom_manip_states.py --scenes D1 Z1
Start plus every manipulation event; start plus scroll events 59,119,...,599.
These are target event states, not smooth_step interpolation/presentation states.
"""
import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from deepzoom_speed import F, memory_status_ex, params, vc, compare


def require_host_memory():
    """Try an actual CIM query; fall back only on query failure, never low memory."""
    error = None
    try:
        query = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreeVirtualMemory"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
        kb = int(query.stdout.strip())
        if kb < 0:
            raise ValueError(f"invalid CIM FreeVirtualMemory: {kb}")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        error = str(exc)
        kb = memory_status_ex()
    if kb < 8_388_608:
        raise RuntimeError(f"FreeVirtualMemory {kb} KB < 8388608 KB")
    return {"query": "GlobalMemoryStatusEx" if error else "CIM",
            "available_virtual_kb": kb, "minimum_kb": 8_388_608,
            "cim_error": error}


def script_states(data, kind):
    """Event indices are zero based, -1 is the start. No eps exponent shortcut."""
    state = tuple(float(v) for v in data["start"])
    yield -1, state
    for index, event in enumerate(data["events"]):
        state = params.apply_event(state, event)
        if kind == "manip" or (index + 1) % 60 == 0:
            yield index, state


def summarize(rows):
    times = [r["gpu_median_ms"] for r in rows if r["gpu_median_ms"] is not None]
    return {
        "states": len(rows),
        "precision_counts": {p: sum(r["precision"] == p for r in rows) for p in ("f32", "f64")},
        "coordinate_fail_states": [r["event"] for r in rows if not r["coordinates"]["ok"]],
        "block_fail_states": [r["event"] for r in rows if not r["blocks"]["ok"]],
        "max_block_share": max(r["blocks"]["block_share"] for r in rows),
        "mean_mismatch": statistics.mean(r["mismatch"] for r in rows),
        "max_mismatch": max(r["mismatch"] for r in rows),
        "mismatch_fail_states": [r["event"] for r in rows if r["mismatch"] > 0.15],
        "both_classes_fail_states": [r["event"] for r in rows if not r["both_classes_ok"]],
        "timed_states": len(times), "gpu_median_range_ms": [min(times), max(times)] if times else None,
        "gpu_ranges_by_precision_ms": {
            p: [min(t), max(t)] if (t := [r["gpu_median_ms"] for r in rows
                 if r["precision"] == p and r["gpu_median_ms"] is not None]) else None
            for p in ("f32", "f64")},
    }


def write_results(out, result):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    lines = ["# Exact event-state observations", "",
             "Start = event -1. CUDA timings exclude coordinates/uploads/readback; warm-up + median of 3.",
             "", "| Script | States | f32/f64 | Block failures | Max blocks | Mean/max mismatch | GPU median range ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in result["summaries"].items():
        count = s["precision_counts"]
        lines.append(f"| {name} | {s['states']} | {count['f32']}/{count['f64']} | "
                     f"{len(s['block_fail_states'])} | {100*s['max_block_share']:.6f}% | "
                     f"{100*s['mean_mismatch']:.6f}%/{100*s['max_mismatch']:.6f}% | "
                     f"{s['gpu_median_range_ms']} |")
    lines += ["", f"Complete requested subset: {result['complete']}. "
              f"Requested scenes: {', '.join(result['requested_scenes'])}.",
              "Cannot observe smooth interpolation states, window pixels, upload/draw/presentation time or arbitrary centres."]
    out.with_suffix(".summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", choices=["D1", "Z1", "DZa", "DZb"],
                    default=["D1", "Z1", "DZa", "DZb"])
    ap.add_argument("--out", type=Path, default=F / "logs/deepzoom/manip_states.json")
    args = ap.parse_args()
    memory = require_host_memory()  # before importing/initializing CUDA
    print("host memory", memory, flush=True)
    import fast
    scripts = {f"{s}_{k}": F / f"viewer/scripts/{s}_{k}.json"
               for s in args.scenes for k in ("manip", "scroll")}
    sources = ["viewer/fast.py", "viewer/fast_kernel.cu", "mytet_cuda/kernel.cu",
               "viewer/params.py", "tests/deepzoom_blocks.py", "tests/viewer_check.py",
               "bench/deepzoom_manip_states.py", "bench/deepzoom_speed.py"]
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "memory_guard": memory, "device": fast.gpu.device_info(),
              "requested_scenes": args.scenes, "complete": False,
              "shape": [1920, 1080], "max_iter": 500, "escape_radius": 1e10,
              "selection_rule": {"min_relative_spacing": fast.F32_MIN_REL_SPACING,
                                 "min_scale": fast.F32_MIN_SCALE, "max_extent": fast.F32_C_MAX},
              "kernel_options": {"f32": fast.OPTIONS, "f64": fast.F64_OPTIONS},
              "source_sha256": {n: hashlib.sha256((F/n).read_bytes()).hexdigest() for n in sources},
              "script_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in scripts.items()},
              "timing": "every 10th manipulation state (ordinal 0,10,...), all 11 scroll states; warm-up + median of 3",
              "scripts": {}, "summaries": {}}
    begin = time.perf_counter()
    for name, path in scripts.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        kind = name.rsplit("_", 1)[1]
        rows, warmed = [], set()
        for ordinal, (event, state) in enumerate(script_states(data, kind)):
            precision = fast.choose_precision(*state)
            fm, coordinates = fast.fast_map(*state, precision=precision, return_coordinates=True)
            cc = vc.coordinate_counts(coordinates)
            del coordinates
            pm = fast.precise_map(*state)
            timings = []
            if ordinal % 10 == 0 or kind == "scroll":
                if precision not in warmed:
                    fast.fast_map(*state, precision=precision)
                    warmed.add(precision)
                for _ in range(3):
                    timing = {}
                    fast.fast_map(*state, precision=precision, timing=timing)
                    timings.append(timing["gpu_ms"])
            blocks = compare(fm, pm)
            mismatch = float((fm != pm).mean())
            precise_both = bool(pm.any() and (~pm).any())
            fast_both = bool(fm.any() and (~fm).any())
            row = {"ordinal": ordinal, "event": event, "state": state,
                   "state_hex": [v.hex() for v in state], "precision": precision,
                   "coordinates": cc, "blocks": blocks, "mismatch": mismatch,
                   "precise_both_classes": precise_both, "fast_both_classes": fast_both,
                   "both_classes_ok": not precise_both or fast_both,
                   "fast_divergent_pixels": int(fm.sum()), "precise_divergent_pixels": int(pm.sum()),
                   "fast_sha256": hashlib.sha256(fm.tobytes()).hexdigest(),
                   "precise_sha256": hashlib.sha256(pm.tobytes()).hexdigest(),
                   "condition_3a": cc["ok"], "condition_3b": blocks["ok"],
                   "condition_4_per_frame": mismatch <= 0.15 and (not precise_both or fast_both),
                   "gpu_ms": timings, "gpu_median_ms": statistics.median(timings) if timings else None}
            rows.append(row)
            if ordinal % 50 == 0:
                print(name, ordinal, precision, "blocks", blocks["block_share"], "mismatch", mismatch, flush=True)
                result["scripts"][name] = rows
                result["summaries"][name] = summarize(rows)
                write_results(args.out, result)
        result["scripts"][name] = rows
        result["summaries"][name] = summarize(rows)
        print(name, result["summaries"][name], flush=True)
        write_results(args.out, result)
    result["complete"] = True
    result["wall_s"] = time.perf_counter() - begin
    result["pool_bytes"] = fast.pool_bytes()
    result["state_failure_count"] = sum(not (r["condition_3a"] and r["condition_3b"]
                                             and r["condition_4_per_frame"])
                                         for rows in result["scripts"].values() for r in rows)
    result["mean_mismatch_fail_scripts"] = [n for n, s in result["summaries"].items()
                                              if s["mean_mismatch"] > 0.10]
    write_results(args.out, result)
    return int(bool(result["state_failure_count"] or result["mean_mismatch_fail_scripts"]))


if __name__ == "__main__":
    raise SystemExit(main())
