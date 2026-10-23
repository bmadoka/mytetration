"""Hybrid and deep zoom measurement: four scenes x manip/scroll x three new viewer processes.
Run 1 is dropped for fps; runs 2 and 3 are judged independently for coordinates,
blocks, quality and script-specific speed. Original integrity and 2s transitions
are checked on all runs. All timing finishes before GPU diagnostic reruns.

usage: python viewer/measure.py [--out logs/hybrid/measure_<time>]
"""
import argparse
import datetime as dt
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path[:0] = [str(HERE), str(F / "tests"), str(F)]
import launch  # noqa: E402
from cpu_guard import guarded_run

SCENES = ("D1", "Z1", "DZa", "DZb")
SCRIPT_KINDS = ("manip", "scroll")


def integrity(r):
    """Every mandatory invariant of one run (any run, judged or not)."""
    pc = r.get("precise_checks", [])
    return bool(not r["param_mismatch"] and not r["started_before_input"] and r["calls_equal_counted"]
                and r.get("sha_rechecked", 0) == r.get("fresh_count", r["counted"]) and r.get("sha_recheck_failed", [1]) == []
                and r["last_frame_is_precise_of_last_input"]
                and (r.get("precise_analysis_skipped") or (pc
                     and all(all(v for k, v in c.items() if k != "seq") for c in pc)
                     and all("render_target_equal" in c for c in pc)))
                and r["transitions_s"] and max(r["transitions_s"]) <= 2.0
                and r["gpu_use_mib"] <= 2048 and not r["foreground_is_viewer_after_show"]
                and r["events_delivered"] == r["events"] and r.get("present_failed", [1]) == []
                and r.get("window_ok") is True)


def verdict(r, judged, script_kind="manip"):
    """All original invariants plus coordinates and blocks (tests/viewer_check.blocks), quality and the script-specific speed bound."""
    if r.get("contamination",{}).get("verdict",{}).get("contaminated",False):
        return False
    if "hybrid" in r:
        h = r["hybrid"]
        records = h.get("records", {})
        mandatory = (integrity(r) and not h["condition4"]["bad"]
                     and h["condition2"].get("bad_start_params", [1]) == []
                     and h.get("duplicate_jobs", {}).get("count") == 0
                     and all(records.get(k, [1]) == [] for k in ("bad_transform", "bad_bound_source", "bad_order"))
                     and not any(records.values())
                     and (r.get("scene") != "DZb" or h["condition2"]["ok"]))
        if not judged:
            return None if mandatory else False
        speed = h["condition1"]["ok"]
        shallow = (r.get("scene") not in ("D1", "Z1") or script_kind != "manip"
                   or (h["condition4"]["warped"] == 0 and r["fps"] >= 30 and r["frame_time_p95_ms"] <= 50))
        if r.get("analysis_skipped"):
            return None if mandatory and speed and shallow else False
        b, q = r.get("blocks", {}), r.get("quality", {})
        honesty = r.get("honesty", {})
        return bool(mandatory and speed and shallow
                    and honesty.get("expected_frames", 0) > 0
                    and len(honesty.get("frames", [])) == honesty.get("expected_frames")
                    and honesty.get("bad_pixels") == 0 and honesty.get("render_target_bad_pixels") == 0
                    and honesty.get("bound_source_mismatch") == []
                    and r.get("present_blocks", {}).get("frames") == honesty.get("expected_frames")
                    and r["present_blocks"].get("moving_frames") == r["counted"]
                    and not r["present_blocks"]["bad_blocks"] and b.get("frames") == r["counted"]
                    and not any(b.get(k, [1]) for k in ("bad_coordinates", "bad_blocks", "sha_mismatch", "precision_mismatch"))
                    and q.get("frames") == r["counted"] and q["mean"] <= .10 and q["max"] <= .15
                    and not q["both_judgements_missing"])
    if not judged:
        return None if integrity(r) else False
    q = r.get("quality") or {}
    b = r.get("blocks") or {}
    blocks_ok = (b.get("frames") == r["counted"] and not b.get("bad_coordinates", [1])
                 and not b.get("bad_blocks", [1]) and not b.get("sha_mismatch", [1])
                 and not b.get("precision_mismatch", [1]))
    return bool(integrity(r) and blocks_ok and r["fps"] >= 30
                and r["frame_time_p95_ms"] <= (33 if script_kind == "scroll" else 50)
                and q.get("frames") == r["counted"] and q["mean"] <= 0.10 and q["max"] <= 0.15
                and not q["both_judgements_missing"])


@contextmanager
def phase(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: elapsed {time.perf_counter() - start:.3f} s", flush=True)


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def nonnegative(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def arguments(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    ap.add_argument("--kinds", nargs="+", choices=SCRIPT_KINDS, default=list(SCRIPT_KINDS))
    ap.add_argument("--runs", type=positive, default=3, help="total processes per scene/kind; first run dropped")
    ap.add_argument("--no-precise", action="store_true", help="timing/records and fresh SHA only; no precise comparisons or replay")
    ap.add_argument("--no-present-precise", action="store_true", help="legacy: skip C5/6 and block/quality; retain precise stop check")
    ap.add_argument("--blocks-runs", type=nonnegative, default=None,
                    help="C5/6 and quality for first N judged runs per scene/kind; default all")
    return ap.parse_args(argv)


def main(argv=None):
    a = arguments(argv)
    out = Path(a.out or F / "logs" / "hybrid" / f"measure_{dt.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=False)
    runs = {}
    for scene in a.scenes:
        for kind in a.kinds:
            for k in range(1, a.runs + 1):
                key = f"{scene}_{kind}_run{k}"
                with phase(f"timed {key}"):
                    runs[key] = guarded_run(lambda directory: launch.run(
                        F / "viewer/scripts" / f"{scene}_{kind}.json", "timed", directory,readback="precise"),out/key)
                print(f"{dt.datetime.now():%H:%M:%S} timed {key} done", flush=True)
    import viewer_check as vc
    vc.require_memory()
    res, verdicts, transitions = {}, [], {}
    lines = ["| Scene / script / run | Hz | C1 normal fps / p95 ms | Exception count / seconds / fps / p95 ms | C2 gap s / start errors | C3 transition s | C4 mode errors / warped | C5 median / p95 / max % | Deepzoom coordinates / quality | Observation |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, directory in runs.items():
        scene, kind, run = key.split("_")
        judged = run != "run1"
        with phase(f"records/fresh SHA/stop {key}"):
            r = vc.check_run(directory, precise_check=not a.no_precise)
        r["scene"] = scene
        r["contamination"] = json.loads((directory/"contamination.json").read_text(encoding="utf-8"))
        selected = a.blocks_runs is None or int(run[3:]) - 1 <= a.blocks_runs
        if judged and selected and not (a.no_precise or a.no_present_precise):
            with phase(f"source coordinates {key}"):
                r["blocks"] = vc.blocks(directory, check_blocks=False)
            with phase(f"C5/C6/quality replay {key}"):
                replay = vc.check_replay(directory)
                r["honesty"] = replay
                r["quality"] = replay["quality"]
                r["present_blocks"] = replay["condition5"]
        elif judged:
            r["analysis_skipped"] = True
        res[key] = r
        ok = verdict(r, judged, kind)
        verdicts.append(ok is not False)
        h = r["hybrid"]
        speed, replace, mode = h["condition1"], h["condition2"], h["condition4"]
        normal, exc = speed["normal"], speed["exception"]
        block = r.get("present_blocks", {}).get("distribution", {})
        block_text = " / ".join(f"{100 * block[str(q)]:.3f}" for q in (50, 95, 100)) if block else "not measured"
        q, b = r.get("quality"), r.get("blocks")
        old = f"{b['min_row']}/{b['min_column']}; {100*q['mean']:.3f}/{100*q['max']:.3f}%" if q else "not measured"
        trans = r["transitions_s"]
        transitions.setdefault(f"{scene}_{kind}", []).extend(trans)
        lines.append(f"| {key} | {speed['monitor_refresh_hz']} | {normal['fps']:.2f} / {normal['p95_ms']:.2f} | "
                     f"{exc['count']} / {exc['length_s']:.4f} / {exc['fps']:.2f} / {exc['p95_ms']:.2f} | "
                     f"{replace['max_gap_s']:.4f} / {len(replace['bad_start_params'])} | {trans} | "
                     f"{len(mode['bad'])} / {mode['warped']} | {block_text} | {old} | "
                     f"{ok if judged or ok is False else 'first run dropped'}; contaminated={r['contamination']['verdict']['contaminated']} |")
        print(f"{key}: C1 {speed['ok']}, C2 gap {replace['max_gap_s']:.4f}, mode errors {mode['bad']}", flush=True)
    low = [k for k, r in res.items() if r["hybrid"]["condition1"]["below_60_hz"]]
    if low:
        lines[:0] = ["Monitor below 60 Hz: " + ", ".join(low) + ". Limits use 95% refresh and 1.05 refresh periods.", ""]
    lines += ["", "Transition median / max across selected processes (limit 2s):"]
    lines += [f"{key}: {np.median(t):.4f} / {max(t):.4f} s" for key, t in transitions.items() if t]
    lines += ["", f"Run 1 speed/quality dropped; {a.runs} total processes per scene/kind. All runs: fresh SHA, start parameters, modes and transform/source records.",
              "C2 gap judged for DZb; start parameters required in every scene. C5 reads every replay present's full render target; PNGs are display | precise.",
              "--no-precise omits precise equality and C5/6/quality. --blocks-runs limits C5/6/quality to first N judged runs per scene/kind. Skipped analysis stays unmeasured."]
    scene_blocks = {}
    for scene in a.scenes:
        observations = [dict(row, run=key) for key, run in res.items() if run["scene"] == scene
                        for row in run.get("present_blocks", {}).get("per_present", [])]
        shares = [row["block_share"] for row in observations]
        scene_blocks[scene] = {"frames": len(observations),
                               "distribution": {str(q): float(np.percentile(shares, q)) if shares else None
                                                for q in (0, 50, 95, 100)},
                               "worst": sorted(observations, key=lambda r: (-r["block_share"], r["run"], r["seq"]))[:3]}
        if shares:
            lines.append(f"{scene} judged replay presents: n={len(shares)}, median/p95/max blocks "
                         f"{100*np.median(shares):.3f}/{100*np.percentile(shares, 95):.3f}/{100*max(shares):.3f}%.")
    res["_per_scene_blocks"] = scene_blocks
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if all(verdicts) else 1)


if __name__ == "__main__":
    main()
