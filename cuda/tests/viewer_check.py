"""Checks of viewer logs. Used by tests/test_viewer.py and viewer/measure.py.

Independent of the viewer: expected parameters come from the script file + the contract equations written below
(viewer/params.py is pinned to them bit for bit by tests/test_viewer.py); the expected pictures come from the
colouring below (numpy), not from the viewer's shader; the precise reference is mytet_cuda called directly.

Result keys of the hybrid display checks in this file: condition1 = present speed (present_speed), condition2 =
picture replacement gaps (replacement_check), condition4 = fresh-only / warped mode replay (replay_modes),
condition5 (also condition5.json) = block share of every present while moving (honesty replay); condition6.json =
window pixels equal to the independently recomputed picture (honesty replay). They are names, not references to
an outside document.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

F = Path(__file__).resolve().parents[1]
for p in (F / "viewer", F):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


MIN_FREE_VIRTUAL_KB = 8_388_608


class MemoryRefused(RuntimeError):
    pass


def require_memory():
    """Before a large GPU run in this (parent) process: FreeVirtualMemory >= 8,388,608 KB, else refuse
    (memory rule of the script paths; the child viewer checks it separately)."""
    import desktop
    kb = desktop.free_virtual_kb()
    if kb < MIN_FREE_VIRTUAL_KB:
        raise MemoryRefused(f"FreeVirtualMemory {kb} KB < {MIN_FREE_VIRTUAL_KB} KB: large GPU run refused")
    return kb


def picture(divergence_map):
    """Picture rule: render-target R[py, px] (rows bottom->top) white iff map[px, py] (16by9_H origin lower)."""
    m = np.asarray(divergence_map, dtype=np.bool_).T
    out = np.zeros(m.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.where(m, 255, 0).astype(np.uint8)[..., None]
    out[..., 3] = 255
    return out


def window_of(rt):
    """Window rule: window (u, v) = render target (2u, 2v); RGB."""
    return np.ascontiguousarray(rt[::2, ::2, :3])


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load(log_dir):
    d = Path(log_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    version = recording_version(meta)
    frames = [json.loads(line) for line in (d / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    if version >= 2:
        for f in frames:
            fields=CANDIDATE_FIELDS + (("candidates",) if version==3 else ())
            if (f["kind"] == "fast" or version==3) and any(k not in f for k in fields):
                raise ValueError(f"B2 present record error {f['seq']}: missing candidate fields")
        jobs = [json.loads(line) for line in (d / "fresh.jsonl").read_text(encoding="utf-8").splitlines()]
        duplicate_job_check(frames, jobs, version)
    script = json.loads((F / meta["script"]).read_text(encoding="utf-8"))
    if version==3:
        fresh={r["id"]:r for r in jobs}
        load_auxiliary(d,fresh,frames)  # missing/invalid margin and precise data is an error
    return meta, frames, script


def contract_step(state, ev):
    """The move / zoom equations of viewer/README.md section 2, written here again from that text
    (independent of viewer/params.py, which tests/test_viewer.py pins to these equations bit for bit)."""
    x0, y0, eps = state
    if ev["type"] == "drag":
        eps_y = eps * (9 / 16)
        return x0 - ev["dx"] * (2 * eps / 1919), y0 + ev["dy"] * (2 * eps_y / 1079), eps
    s = ev["steps"]
    if s > 0:
        return x0, y0, eps * (0.9 ** s)
    if s < 0:
        return x0, y0, eps * ((1 / 0.9) ** (-s))
    return x0, y0, eps


def expected_params(script, upto):
    """Parameters after script events 0 .. upto (inclusive), from the script file and the contract equations."""
    state = tuple(float(v) for v in script["start"])
    for k, ev in enumerate(script["events"]):
        if upto is not None and k > upto:
            break
        state = contract_step(state, ev)
    return state


def contract_smooth(display, target, dt):
    """Re-encoded from the smooth-zoom rule (viewer/README.md section 4, viewer/params.py docstring), without importing params.smooth_step."""
    if float(display[2]).hex() == float(target[2]).hex():
        return tuple(target)
    lt = math.log(target[2])
    ld = math.log(display[2])
    a = min(0.5, 1 - math.exp(-dt / 0.06))
    ln_new = ld + a * (lt - ld)
    eps = target[2] if abs(lt - ln_new) <= 0.02 * math.log(1 / 0.9) else math.exp(ln_new)
    return target[0], target[1], eps


def expected_displays(meta, frames, script):
    """Independent event replay and S; only logged decision times are S's clock inputs.

    Return expected parameters by sequence and violations of the recorded decision
    contract, including undelivered due events and all hex/display/target aliases.
    """
    expected, bad = {}, []
    previous = tuple(float(v) for v in script["start"])
    previous_t = None
    inputs = {int(k): t for k, t in meta["input_t"].items()}
    for f in frames:
        t = f["t_decide"]
        delivered = max((k for k, ti in inputs.items() if ti <= t), default=-1)
        target = expected_params(script, delivered)
        ok = (f["k"] == f["last_input"] == delivered and t <= f["t_compute_start"])
        if meta["mode"] != "verify":
            due = max((k for k, e in enumerate(script["events"])
                       if meta["t_playback_start"] + e["t"] <= t), default=-1)
            # Initial frame precedes the playback clock.
            if f["kind"] != "initial":
                ok = ok and due == delivered
        if f["kind"] == "fast":
            ok = ok and previous_t is not None and t >= previous_t
            p = contract_smooth(previous, target, t - previous_t)
            want_type = "new_input" if f["t_first_new_input"] is not None else "interp"
            ok = ok and f["frame_type"] == want_type
        else:
            p = target
        hx = [float(v).hex() for v in p]
        ok = (ok and same_bits(f["params"], p) and same_bits(f["display_params"], p)
              and f["params_hex"] == hx and f["display_params_hex"] == hx
              and f["target_params_hex"] == [float(v).hex() for v in target])
        expected[f["seq"]] = p
        if not ok:
            bad.append(f["seq"])
        previous, previous_t = p, t
    return expected, bad


def reference_precise(x0, y0, eps):
    """Precise reference = the existing double implementation with the viewer's fixed settings (16by9_H,
    n = 1920, max_iter 500, escape_radius 1e10), called directly — not through the viewer's wrapper."""
    import mytet_cuda
    return mytet_cuda.divergence_map("16by9_H", x0, y0, eps, 1920, 500, 1e10)


def same_bits(p, q):
    return [float(a).hex() for a in p] == [float(b).hex() for b in q]


def counted_frames(frames):
    """Fast frames after the start whose parameters differ from the frame shown before them."""
    if any("present_kind" in f for f in frames):
        return [f for f in frames if f["kind"] == "fast"]
    out, prev = [], None
    for f in frames:
        if f["kind"] == "fast" and prev is not None and not same_bits(f["params"], prev["params"]):
            out.append(f)
        prev = f
    return out


def contamination_report(log_dir, mode):
    path = Path(log_dir) / "contamination.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if type(report["verdict"]["contaminated"]) is not bool:
            raise ValueError("missing contamination verdict")
        return report
    except (OSError,ValueError,KeyError,TypeError) as error:
        # Older untimed archives can still be inspected. Timed window evidence
        # without a valid guard report is infrastructure failure.
        return {"verdict":{"contaminated":mode in ("timed","media","callbacks"),
                           "errors":[str(error)]}}


def check_run(log_dir, recheck=True, precise_check=True):
    """Everything that can be judged from one run. GPU needed for recheck/precise_check."""
    if recheck or precise_check:
        require_memory()
    meta, frames, script = load(log_dir)
    ev = script["events"]
    input_t = {int(k): v for k, v in meta["input_t"].items()}
    fast = [f for f in frames if f["kind"] == "fast"]
    counted = counted_frames(frames)
    r = {"frames_fast": len(fast), "counted": len(counted), "dropped_same_params": len(fast) - len(counted),
         "skipped_same_param_inputs": meta["skipped_same_params"], "fast_calls": meta["fast_calls_playback"],
         "events": len(ev), "events_delivered": meta["events_delivered"]}
    r["contamination"] = contamination_report(log_dir, meta.get("mode"))
    # Fast = independent display replay; precise = unchanged target replay.
    display_expected, r["param_mismatch"] = expected_displays(meta, frames, script)
    # computation starts after the last reflected input was handled
    r["started_before_input"] = [f["seq"] for f in frames if f["kind"] != "initial" and f["last_input"] >= 0
                                 and f["t_compute_start"] < input_t[f["last_input"]]]
    r["calls_equal_counted"] = (meta["fast_calls_playback"] == len(counted)
                                and meta["precision_calls_playback"] == [f["precision"] for f in fast])
    hybrid_run = recording_version(meta) in (1, 2, 3)
    if hybrid_run:
        fresh, saved = load_fresh(log_dir)
        r["calls_equal_counted"] = (meta["fast_calls_playback"] == len(fresh)
                                    and meta["precision_calls_playback"] == [x["precision"] for x in
                                         sorted(fresh.values(), key=lambda x: x["t_start"])])
    pres = np.array([f["t_present"] for f in counted])
    first_input = min(input_t.values()) if input_t else None
    if len(counted) >= 2:
        gaps = np.diff(pres)
        lat = np.array([f["t_present"] - f["t_first_new_input"] for f in counted
                        if f["frame_type"] == "new_input"])
        r.update({"fps": len(counted) / (pres[-1] - first_input), "frame_time_p95_ms": 1000 * float(np.percentile(gaps, 95)),
                  "frame_time_max_ms": 1000 * float(gaps.max()), "frames_over_50_ms": int((gaps > 0.050).sum()), "frame_time_median_ms": 1000 * float(np.median(gaps)),
                  "latency_median_ms": 1000 * float(np.median(lat)), "latency_p95_ms": 1000 * float(np.percentile(lat, 95))})
    # precise frames: transition time from their last input, expected parameters, map and picture
    precise = [f for f in frames if f["kind"] == "precise"]
    r["transitions_s"] = [f["t_present"] - input_t[f["last_input"]] for f in precise if f["last_input"] >= 0]
    r["precise_frames"] = len(precise)
    r["last_frame_is_precise_of_last_input"] = bool(precise) and frames[-1]["kind"] == "precise" and \
        frames[-1]["last_input"] == len(ev) - 1
    # fast-mode return: a fast frame directly after a precise frame (script with an input after a pause)
    r["fast_after_precise"] = [frames[k + 1]["seq"] for k, f in enumerate(frames[:-1])
                               if f["kind"] == "precise" and frames[k + 1]["kind"] == "fast"]
    if recheck or precise_check:
        sys.path.insert(0, str(F / "viewer"))
        import fast as fastmod
    if recheck:  # Every fresh map, including those never first displayed during motion.
        if hybrid_run:
            bad = [ident for ident, f in fresh.items()
                   if sha(fastmod.fast_map(*f["params"])) != f["sha256"] or sha(saved[ident]) != f["sha256"]]
            r["sha_rechecked"] = len(fresh)
            r["fresh_count"] = len(fresh)
        else:
            bad = [f["seq"] for f in counted if sha(fastmod.fast_map(*f["params"])) != f["map_sha256"]]
            r["sha_rechecked"] = len(counted)
        r["sha_recheck_failed"] = bad
        ret = []
        for seq in r["fast_after_precise"]:
            f = frames[seq]
            if hybrid_run:
                src = fresh[f["source_fresh_id"]]
                ret.append(sha(fastmod.fast_map(*src["params"])) == src["sha256"])
            else:
                ret.append(sha(fastmod.fast_map(*display_expected[f["seq"]])) == f["map_sha256"])
        r["fast_return_sha_ok"] = ret
    r["precise_analysis_skipped"] = not precise_check
    if precise_check:
        pc = []
        for f in precise:
            exp = expected_params(script, f["last_input"])
            m = reference_precise(*exp)
            item = {"seq": f["seq"], "params_bit_equal": same_bits(f["params"], exp),
                    "started_after_last_input": f["t_compute_start"] >= input_t[f["last_input"]],
                    "map_equal": sha(m) == f["map_sha256"]}
            if "rt_sha256" in f:
                item["render_target_equal"] = sha(picture(m)) == f["rt_sha256"]
            if "bb_sha256" in f:
                item["back_buffer_equal"] = sha(window_of(picture(m))) == f["bb_sha256"]
            pc.append(item)
        r["precise_checks"] = pc
    # presentation: every frame's DwmFlush succeeded; the window was shown, not minimised, topmost, never activated
    r["present_failed"] = [f["seq"] for f in frames if f.get("present_hr") != 0]
    ws = [meta.get("window_state_after_show", {}), meta.get("window_state_at_end", {})]
    r["window_ok"] = bool(meta.get("topmost_set") is True and all(
        s.get("visible") is True and s.get("minimised") is False and s.get("topmost") is True
        and s.get("foreground_is_viewer") is False for s in ws))
    r["gpu_use_mib"] = (meta["pool_max_bytes"] / 2 ** 20) + meta["cuda_context_mib"]
    r["gl_context_mib"] = meta["gl_context_mib"]
    r["foreground_is_viewer_after_show"] = meta["foreground_is_viewer_after_show"]
    if hybrid_run:
        r["hybrid"] = check_hybrid_records(meta, frames, list(fresh.values()))
    return r


def check_display(log_dir):
    """Verify replay (readback of every frame): render target = picture of the frame's map, window = rule-reduced
    render target. The frame's map is recomputed from its logged parameters (fast or precise function)."""
    require_memory()
    sys.path.insert(0, str(F / "viewer"))
    import fast as fastmod
    meta, frames, script = load(log_dir)
    fresh, maps = load_fresh(log_dir) if recording_version(meta) in (1, 2, 3) else ({}, {})
    bad_rt, bad_bb, n, n_bb, missing = [], [], 0, 0, []
    for f in frames:
        if "rt_sha256" not in f or "bb_sha256" not in f:
            missing.append(f["seq"])
        if "rt_sha256" not in f:
            continue
        m = (displayed_map(f, fresh, maps, meta) if fresh and f["kind"] == "fast" else
             fastmod.fast_map(*f["params"]) if f["kind"] == "fast" else reference_precise(*f["params"]))
        if "map_sha256" in f and f.get("present_kind")!="warped" and sha(m) != f["map_sha256"]:
            bad_rt.append((f["seq"], "map"))
            continue
        pic = picture(m)
        n += 1
        if sha(pic) != f["rt_sha256"]:
            bad_rt.append((f["seq"], "rt"))
        if "bb_sha256" in f:
            n_bb += 1
            if sha(window_of(pic)) != f["bb_sha256"]:
                bad_bb.append(f["seq"])
    return {"frames_checked": n, "back_buffer_checked": n_bb, "frames_missing_readback": missing,
            "render_target_mismatch": bad_rt, "back_buffer_mismatch": bad_bb,
            "fast_frames": sum(f["kind"] == "fast" for f in frames), "frames": len(frames)}


def quality(log_dir):
    """Fast map vs precise map (existing double implementation) for every counted frame."""
    require_memory()
    sys.path.insert(0, str(F / "viewer"))
    import fast as fastmod
    meta, frames, script = load(log_dir)
    fresh, maps = load_fresh(log_dir) if recording_version(meta) in (1, 2, 3) else ({}, {})
    ratios, both_bad = [], []
    for f in counted_frames(frames):
        fm = displayed_map(f, fresh, maps, meta) if fresh else fastmod.fast_map(*f["params"])
        pm = reference_precise(*f["params"])
        ratios.append(float((fm != pm).mean()))
        if pm.any() and (~pm).any() and not (fm.any() and (~fm).any()):
            both_bad.append(f["seq"])
    return {"frames": len(ratios), "mean": float(np.mean(ratios)) if ratios else None,
            "max": float(np.max(ratios)) if ratios else None, "both_judgements_missing": both_bad}


def coordinate_counts(c):
    """Actual kernel c[px,py,(real,imag)]: all x on each row, all y on each column.

    Count the returned real coordinates along each row and imaginary coordinates
    along each column; never infer counts from the input axes alone.
    """
    if isinstance(c, dict):
        ref, gamma = c["reference"], c["gamma"]
        if ref.shape != (2,) or ref.dtype != np.float64 or gamma.dtype != np.float32:
            raise ValueError("unexpected pt32 coordinate representation")
        result = coordinate_counts(gamma)
        result["representation"] = "float64 reference + float32 gamma"
        result["finite"] = bool(result["finite"] and np.isfinite(ref).all())
        result["ok"] = bool(result["ok"] and result["finite"])
        return result
    if c.shape != (1920, 1080, 2) or c.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"unexpected kernel coordinates {c.shape} {c.dtype}")
    # c is separable in the two implementations, so vectorized scalar uniqueness
    # gives exact counts for all 1080 rows and 1920 columns without Python sorting.
    sx = np.sort(c[..., 0], axis=0)
    sy = np.sort(c[..., 1], axis=1)
    rows = 1 + np.count_nonzero(sx[1:] != sx[:-1], axis=0)
    cols = 1 + np.count_nonzero(sy[:, 1:] != sy[:, :-1], axis=1)
    return {"min_row": int(rows.min()), "min_column": int(cols.min()),
            "finite": bool(np.isfinite(c).all()),
            "ok": bool(np.isfinite(c).all() and np.all(rows == 1920) and np.all(cols == 1080))}


def blocks(log_dir, check_blocks=True):
    """Full-resolution coordinates (a) and blocks (b) of every counted frame, using its chosen map kernel.

    Recomputes the full frame plus actual orbit coordinate output; sha links this
    diagnostic rerun to the logged frame. No replacement with host grid counts.
    """
    require_memory()
    import fast as fastmod
    from deepzoom_blocks import compare
    meta, frames, script = load(log_dir)
    fresh, maps = load_fresh(log_dir) if recording_version(meta) in (1, 2, 3) else ({}, {})
    rows, bad_coordinates, bad_blocks, bad_sha, bad_precision = [], [], [], [], []
    source_cache = {}
    for f in counted_frames(frames):
        src = fresh[f["source_fresh_id"]] if fresh else f
        ident = src.get("id", src.get("seq"))
        if ident not in source_cache:
            raw, c = fastmod.fast_map(*src["params"], return_coordinates=True)
            cc = coordinate_counts(c)
            del c
            source_cache[ident] = (cc, sha(raw), fastmod.calls["precision"][-1])
        cc, source_sha, precision = source_cache[ident]
        if f["precision"] != precision:
            bad_precision.append(f["seq"])
        if not cc["ok"]:
            bad_coordinates.append(f["seq"])
        if source_sha != src.get("sha256", src.get("map_sha256")):
            bad_sha.append(f["seq"])
        block = {}
        if check_blocks:
            fm = displayed_map(f, fresh, maps, meta) if fresh else raw
            block = compare(fm, reference_precise(*f["params"]))
            if not block["ok"]:
                bad_blocks.append(f["seq"])
        rows.append({"seq": f["seq"], "precision": precision, "coordinates": cc, **block})
    result = {"frames": len(rows), "bad_coordinates": bad_coordinates, "bad_blocks": bad_blocks,
              "sha_mismatch": bad_sha, "precision_mismatch": bad_precision, "per_frame": rows,
              "min_row": min((r["coordinates"]["min_row"] for r in rows), default=0),
              "min_column": min((r["coordinates"]["min_column"] for r in rows), default=0),
              "max_block_share": max((r.get("block_share", 0.0) for r in rows), default=0.0),
              "block_analysis_skipped": not check_blocks}
    (Path(log_dir) / "blocks.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def one_notch(log_dir):
    """Observations of one wheel notch (zoom steps and settle time); assertions belong to the replay test body."""
    meta, frames, script = load(log_dir)
    fast = [f for f in frames if f["kind"] == "fast"]
    baseline = next(f for f in frames if f["kind"] == "initial")
    seq = [baseline] + fast
    delta = [abs(math.log(b["params"][2]) - math.log(a["params"][2])) for a, b in zip(seq, seq[1:])]
    changed = sum(float(a["params"][2]).hex() != float(b["params"][2]).hex() for a, b in zip(seq, seq[1:]))
    last_input = meta["input_t"][str(len(script["events"]) - 1)]
    target = expected_params(script, len(script["events"]) - 1)
    snapped = [f["t_present"] - last_input for f in fast if same_bits(f["params"], target)]
    return {"max_dlog": max(delta, default=0.0), "changed": changed,
            "snap_s": min(snapped) if snapped else None,
            "frames": [{"t": f["t_present"] - last_input, "t_decide": f["t_decide"] - last_input,
                        "eps": f["params"][2], "eps_hex": float(f["params"][2]).hex()} for f in seq]}


class PackedFreshMaps:
    """Unpack only the requested source: long shallow runs need not hold GB of bool maps."""
    def __init__(self, bits, ids):
        self.bits = bits
        self.rows = {ident: row for row, ident in enumerate(ids)}

    def __getitem__(self, ident):
        return np.unpackbits(self.bits[self.rows[ident]])[:1920 * 1080].reshape(1920, 1080).astype(bool)


def load_fresh(log_dir):
    d = Path(log_dir)
    rows = [json.loads(s) for s in (d / "fresh.jsonl").read_text(encoding="utf-8").splitlines()]
    bits = np.load(d / "fresh_maps.npy", mmap_mode="r", allow_pickle=False)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or bits.shape != (len(rows), 259200) or bits.dtype != np.uint8:
        raise ValueError("invalid packed fresh archive")
    maps=PackedFreshMaps(bits,ids)
    frame_path=d/"frames.jsonl"
    maps.precise_params={f["precise_id"]:f["display_params"] for f in
        (json.loads(line) for line in (frame_path.read_text(encoding="utf-8").splitlines() if frame_path.exists() else [])) if "precise_id" in f}
    meta_path=d/"meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")).get("hybrid_version")==3:
        frames=[json.loads(line) for line in frame_path.read_text(encoding="utf-8").splitlines()]
        maps.margins,maps.precise_maps=load_auxiliary(d,{r["id"]:r for r in rows},frames)
    return {r["id"]: r for r in rows}, maps


def independent_warp(source_map, display, source):
    """Independent numpy transcription of the hybrid display rule. No app indices or GPU functions."""
    xd, yd, ed = [float(v) for v in display]
    xs, ys, es = [float(v) for v in source]
    eyd, eys = ed * (9 / 16), es * (9 / 16)
    xx = ((xd - xs) - ed + es + np.arange(1920, dtype=np.float64) * (2 * ed / 1919)) / (2 * es / 1919)
    yy = ((yd - ys) - eyd + eys + np.arange(1080, dtype=np.float64) * (2 * eyd / 1079)) / (2 * eys / 1079)
    xx = np.maximum(0, np.minimum(1919, np.floor(xx + 0.5))).astype(np.int64)
    yy = np.maximum(0, np.minimum(1079, np.floor(yy + 0.5))).astype(np.int64)
    return np.asarray(source_map, dtype=bool)[xx[:, None], yy[None, :]]


def independent_candidates(frame, fresh, maps):
    """Independent hybrid-rule NumPy compositor, driven only by saved records/maps."""
    xd, yd, ed = map(float, frame["display_params"])
    ids = frame["candidate_ids"]
    remaining = np.ones((1920, 1080), bool)
    result = np.empty_like(remaining)
    counts = {"newest": 0, "older": 0, "edge": 0}
    fallback = None
    for n, ident in enumerate(ids):
        xs, ys, es = map(float, fresh[ident]["params"])
        ey, sy = ed * (9 / 16), es * (9 / 16)
        dx, dy = (2 * es / 1919), (2 * sy / 1079)
        ox, oy = ((xd-xs)-ed+es)/dx, ((yd-ys)-ey+sy)/dy
        x = ((xd-xs)-ed+es+np.arange(1920, dtype=np.float64)*(2*ed/1919))/dx
        y = ((yd-ys)-ey+sy+np.arange(1080, dtype=np.float64)*(2*ey/1079))/dy
        tr = frame.get("candidate_transforms")
        if tr is not None:
            x += (float(tr[n]["offset"][0])-ox) + np.arange(1920)*(float(tr[n]["scale"])-ed/es)
            y += (float(tr[n]["offset"][1])-oy) + np.arange(1080)*(float(tr[n]["scale"])-ed/es)
        x, y = np.floor(x+.5), np.floor(y+.5)
        cover = ((x >= 0) & (x < 1920))[:, None] & ((y >= 0) & (y < 1080))[None, :]
        sampled = maps[ident][np.clip(x, 0, 1919).astype(np.int64)[:, None],
                               np.clip(y, 0, 1079).astype(np.int64)[None, :]]
        if n == 0:
            fallback = sampled
        take = remaining & cover
        result[take] = sampled[take]
        counts["newest" if n == 0 else "older"] += int(take.sum())
        remaining &= ~cover
    result[remaining] = fallback[remaining]
    counts["edge"] = int(remaining.sum())
    return result, counts


def displayed_map(frame, fresh, maps, meta=None, *, margins=None, precise_maps=None):
    if meta is not None and recording_version(meta)==3:
        return independent_b3(frame,fresh,maps,margins if margins is not None else getattr(maps,"margins",{}),
                              precise_maps if precise_maps is not None else getattr(maps,"precise_maps",{}),
                              precise_params=getattr(maps,"precise_params",{}))[0]
    src = fresh[frame["source_fresh_id"]]
    if frame["present_kind"] == "warped" and is_b2_recording([frame], meta):
        return independent_candidates(frame, fresh, maps)[0]
    m = maps[src["id"]]
    return independent_warp(m, frame["display_params"], src["params"]) if frame["present_kind"] == "warped" else m


def replay_modes(frames, fresh):
    """Independent mode replay (key condition4): only finished records at that present's decision."""
    import statistics
    mode, rows, bad = "fresh-only", [], []
    for f in frames:
        t = f["t_mode_decide"]
        available = sorted((r for r in fresh if r["t_end"] <= t), key=lambda r: r["t_end"])
        durations = [r["t_end"] - r["t_start"] for r in available[-8:]]
        m = statistics.median(durations) if durations else None
        if m is None:
            mode = "fresh-only"
        elif mode == "fresh-only" and m > 0.033:
            mode = "warped"
        elif mode == "warped" and m <= 0.028:
            mode = "fresh-only"
        valid = f.get("mode") == mode and f.get("m") == m and f["t_decide"] <= t <= f["t_present"]
        src = next((r for r in available if r["id"] == f["source_fresh_id"]), None)
        if f["kind"] != "fast":
            if not valid:
                bad.append(f["seq"])
            rows.append({"seq": f["seq"], "mode": mode, "m": m})
            continue
        if src is None or not available or src["id"] != available[-1]["id"]:
            valid = False
        if mode == "fresh-only":
            valid = (valid and f.get("present_kind") == "fresh" and src is not None and not src.get("async")
                     and src["t_decide"] == f["t_decide"] and same_bits(src["params"], f["params"]))
        else:
            valid = valid and f.get("present_kind") == "warped"
        if not valid:
            bad.append(f["seq"])
        rows.append({"seq": f["seq"], "mode": mode, "m": m})
    return {"per_present": rows, "bad": bad, "warped": sum(f.get("present_kind") == "warped" for f in frames)}


def replacement_check(meta, frames, fresh):
    """Replacement gaps (key condition2) include both boundary gaps, and checks each fresh start decision."""
    times = list(meta["input_t"].values())
    first, last = (min(times), max(times)) if times else (None, None)
    seen, changes, previous = set(), [], None
    by_id = {r["id"]: r for r in fresh}
    for f in frames:
        if f["kind"] != "fast":
            continue
        ident = f["source_fresh_id"]
        if ident not in seen:
            seen.add(ident)
            p = by_id[ident]["params"]
            if previous is None or not same_bits(previous, p):
                previous = p
                if first is not None and first <= f["t_present"] <= last:
                    changes.append(f["t_present"])
    series = [first] + changes + [last] if first is not None else []
    gaps = [b - a for a, b in zip(series, series[1:])]
    bad = []
    decisions = sorted(frames, key=lambda f: f["t_decide"])
    for r in fresh:
        prev = [f for f in decisions if f["t_decide"] <= r["t_start"]]
        f = prev[-1] if prev else None
        hx = [float(v).hex() for v in r["params"]]
        if (f is None or r["t_decide"] != f["t_decide"] or r["k"] != f["k"]
                or not same_bits(r["params"], f["display_params"]) or r["params_hex"] != hx
                or r["display_params_hex"] != hx or r["t_end"] < r["t_start"]):
            bad.append(r["id"])
    max_gap = max(gaps, default=0.0)
    return {"times": series, "max_gap_s": max_gap, "replacements": len(changes), "bad_start_params": bad,
            "ok": bool(first is not None and max_gap <= 0.25 and not bad)}


def duplicate_job_check(frames, fresh, version=2):
    """Recount async starts using present modes; incomplete/inconsistent jobs are record errors."""
    required = {"id", "async", "mode", "params", "t_start", "t_end"}
    for r in fresh if version >= 2 else []:
        try:
            if required - r.keys() or type(r["async"]) is not bool or r["mode"] not in ("warped", "fresh-only"):
                raise ValueError("missing or invalid fields")
            if (len(r["params"]) != 3 or not all(math.isfinite(float(v)) for v in r["params"])
                    or not all(math.isfinite(r[k]) for k in ("t_start", "t_end")) or r["t_end"] < r["t_start"]):
                raise ValueError("invalid params/lifetime")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"job record error {r.get('id')}: {exc}") from exc
    if version >= 2:
        # A warped switching present may have computed synchronously before its mode decision.
        synchronous = [f for f in frames if f.get("kind") == "fast"
                       and ("map_sha256" in f
                            or f.get("t_compute_end", 0) > f.get("t_compute_start", 0))]
        owners = [[] for _ in synchronous]
        for r in fresh:
            matches = [i for i, f in enumerate(synchronous)
                       if same_bits(r["params"], f["params"])
                       and r["t_start"] == f["t_compute_start"] and r["t_end"] == f["t_compute_end"]
                       and ("sha256" not in r or "map_sha256" not in f
                            or r["sha256"] == f["map_sha256"])]
            if (not r["async"] and len(matches) != 1) or (r["async"] and matches):
                raise ValueError(f"job record error {r['id']}: synchronous ownership disagreement")
            for i in matches:
                owners[i].append(r["id"])
        for f, jobs in zip(synchronous, owners):
            if len(jobs) != 1:
                raise ValueError(f"job record error: present {f.get('seq')} requires exactly one synchronous job")
    duplicates = []
    decisions = sorted(frames, key=lambda f: f["t_decide"])
    for r in sorted(fresh, key=lambda r: (r["t_start"], r["id"])):
        if not r.get("async"):
            continue
        prior = [f for f in decisions if f["t_decide"] <= r["t_start"]]
        present = prior[-1] if prior else None
        mode = present.get("mode") if present is not None else None
        if version == 1:
            mode = r.get("mode", mode)  # Historical jobs did not record their launch mode.
        elif (mode not in ("warped", "fresh-only") or r["mode"] != mode
                or ("t_decide" in r and r["t_decide"] != present["t_decide"])):
            raise ValueError(f"job record error {r['id']}: launch mode/present disagreement")
        if mode != "warped":
            continue
        t = r["t_start"]
        computing = [a for a in fresh if a["id"] != r["id"] and
                     (a["t_start"], a["id"]) < (t, r["id"]) and t < a["t_end"]]
        ended = [a for a in fresh if a["id"] != r["id"] and a["t_end"] <= t]
        latest = max(ended, key=lambda a: (a["t_end"], a["id"]), default=None)
        matches = [a["id"] for a in computing if same_bits(a["params"], r["params"])]
        if latest is not None and same_bits(latest["params"], r["params"]):
            matches.append(latest["id"])
        if matches:
            duplicates.append({"id": r["id"], "matches": sorted(set(matches)), "t_start": t})
    return {"count": len(duplicates), "jobs": duplicates}


def present_speed(meta, frames, modes):
    """Present speed (key condition1). Interval belongs to its ending present; include startup in fps duration.

    Exception classification uses independently recomputed modes, so hybrid-off
    cannot turn a slow deep run into an allowed exception by mislabelling it.
    p95 uses consecutive present intervals, as in the original viewer contract.
    """
    fast = [f for f in frames if f["kind"] == "fast"]
    inputs = list(meta["input_t"].values())
    hz = meta.get("monitor_refresh_hz")
    valid_hz = hz is not None and hz > 0
    low = bool(valid_hz and hz < 60)
    fps_limit, p95_limit = (0.95 * hz, 1050 / hz) if low else (60.0, 17.0)
    by_seq = {r["seq"]: r for r in modes["per_present"]}
    previous = min(inputs) if inputs else None
    normal, exception, spans = [], [], []
    for i, f in enumerate(fast):
        if previous is None:
            break
        dt = f["t_present"] - previous
        rule = by_seq[f["seq"]]
        exc = rule["mode"] == "fresh-only" and rule["m"] is not None and rule["m"] > 0.016
        (exception if exc else normal).append((dt, i > 0))
        if exc:
            spans.append({"seq": f["seq"], "start": previous, "end": f["t_present"], "length_s": dt})
        previous = f["t_present"]
    def stats(data, fps_min, p95_max):
        duration = sum(d for d, _ in data)
        gaps = [d for d, interval in data if interval]
        fps = len(data) / duration if duration > 0 else 0.0
        p95 = float(np.percentile(gaps, 95)) * 1000 if gaps else 0.0
        return {"count": len(data), "length_s": duration, "fps": fps, "p95_ms": p95,
                "ok": bool(not data or (duration > 0 and fps >= fps_min and p95 <= p95_max))}
    n, e = stats(normal, fps_limit, p95_limit), stats(exception, 30.0, 50.0)
    return {"normal": n, "exception": e, "exception_intervals": spans, "monitor_refresh_hz": hz,
            "below_60_hz": low, "fps_limit": fps_limit, "p95_limit_ms": p95_limit,
            "ok": bool(fast and inputs and valid_hz and n["ok"] and e["ok"])}


def check_hybrid_records(meta, frames, fresh):
    modes = replay_modes(frames, fresh)
    replacement = replacement_check(meta, frames, fresh)
    speed = present_speed(meta, frames, modes)
    return {"condition1": speed, "condition2": replacement, "condition4": modes,
            "duplicate_jobs": duplicate_job_check(frames, fresh, recording_version(meta)),
            "records": validate_present_records(frames, {r["id"]: r for r in fresh}, meta)}


CANDIDATE_FIELDS = ("candidate_ids", "candidate_transforms", "pixel_counts", "bound_candidate_ids")


def recording_version(meta):
    version = meta.get("hybrid_version")
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError(f"record error: missing or unknown hybrid_version {version!r}")
    return version


def is_b2_recording(frames, meta=None):
    if meta is not None:
        return recording_version(meta) >= 2
    # In-memory draw checks have no metadata. File entry points validate it in load().
    return any(any(k in f for k in CANDIDATE_FIELDS) for f in frames)


def validate_present_records(frames, fresh, meta=None):
    """Independent transform, draw-bound source and recording-wide B2 contract checks."""
    if meta is not None and recording_version(meta)==3:
        return validate_b3(frames,fresh)
    b2 = is_b2_recording(frames, meta)
    bad_transform, bad_bound, bad_order = [], [], []
    previous_t = -math.inf
    first_shown = {}
    for index, f in enumerate(frames):
        if f["seq"] != index or f["t_present"] < previous_t:
            bad_order.append(f["seq"])
        previous_t = f["t_present"]
        if f["kind"] == "fast":
            first_shown.setdefault(f["source_fresh_id"], f["seq"])
        if b2 and f["kind"] == "fast":
            try:
                want_ids = sorted(first_shown, key=lambda i: (first_shown[i], i), reverse=True)[:8]
                ids = f["candidate_ids"]
                if ids != want_ids or not ids or ids[0] != f["source_fresh_id"]:
                    raise ValueError("candidate order")
                if f.get("bound_candidate_ids") != ids:
                    bad_bound.append(f["seq"])
                tr = f["candidate_transforms"]
                if len(tr) != len(ids) or [r["id"] for r in tr] != ids:
                    raise ValueError("candidate transforms")
                xd, yd, ed = map(float, f["display_params"])
                for ident, row in zip(ids, tr):
                    xs, ys, es = map(float, fresh[ident]["params"])
                    ey, sy = ed*(9/16), es*(9/16)
                    want = (ed/es, ((xd-xs)-ed+es)/(2*es/1919), ((yd-ys)-ey+sy)/(2*sy/1079))
                    if not same_bits((row["scale"], *row["offset"]), want):
                        raise ValueError("candidate transform differs")
                # Coverage has no dependency on source values; independent CPU rule checks counts too.
                zero = np.zeros((1920, 1080), bool)
                _, counts = independent_candidates(f, fresh, {i: zero for i in ids})
                if f["pixel_counts"] != counts:
                    raise ValueError("candidate counts")
            except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
                bad_transform.append(f["seq"])
        if "bound_source_fresh_id" not in f or f["bound_source_fresh_id"] != f["source_fresh_id"]:
            bad_bound.append(f["seq"])
        try:
            if f["present_kind"] == "warped":
                xd, yd, ed = map(float, f["display_params"])
                xs, ys, es = map(float, fresh[f["source_fresh_id"]]["params"])
                ey, sy = ed * (9 / 16), es * (9 / 16)
                want = (ed / es, ((xd - xs) - ed + es) / (2 * es / 1919),
                        ((yd - ys) - ey + sy) / (2 * sy / 1079))
            else:
                want = (1., 0., 0.)
                if f["kind"] == "fast" and not same_bits(f["display_params"], fresh[f["source_fresh_id"]]["params"]):
                    raise ValueError("fresh display/source differ")
            got = (f["scale"], *f["offset"])
            if len(got) != 3 or not all(math.isfinite(float(v)) for v in got) or not same_bits(got, want):
                bad_transform.append(f["seq"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            bad_transform.append(f["seq"])
    return {"bad_transform": sorted(set(bad_transform)), "bad_bound_source": sorted(set(bad_bound)), "bad_order": bad_order}


class PreciseCache:
    """One check owns this packed cache, keyed by bit-exact display parameters."""
    def __init__(self, compute=None):
        self.compute = compute or reference_precise
        self.maps = {}

    def __call__(self, *p):
        key = tuple(float(v).hex() for v in p)
        if key not in self.maps:
            self.maps[key] = np.packbits(self.compute(*p))
        return np.unpackbits(self.maps[key])[:1920 * 1080].reshape(1920, 1080).astype(bool)


class ReplayChecks:
    """Stream actual GL readbacks; retain only three worst full-target PNG pairs.

    Fixed RT decoding: R[py,px,0] > 127 -> map[px,py] divergence. Every
    present is checked at 1920x1080; C6 independently checks RGB and alpha.
    """
    def __init__(self):
        self.blocks, self.honesty, self.worst, self.ratios, self.both_bad = [], [], [], [], []

    def add(self, f, rt, bb, expected, pm, bound_id):
        from deepzoom_blocks import compare
        if rt.shape != (1080, 1920, 4) or rt.dtype != np.uint8 or bb.shape != (540, 960, 3) or bb.dtype != np.uint8:
            raise ValueError("replay requires full 1920x1080 RGBA target and 960x540 RGB window")
        shown = (rt[..., 0] > 127).T
        row = {"seq": f["seq"], "kind": f["present_kind"], **compare(shown, pm)}
        self.blocks.append(row)
        if len(self.worst) < 3 or row["block_share"] > self.worst[-1][0]["block_share"]:
            self.worst.append((row, rt.copy(), picture(pm)))
            self.worst.sort(key=lambda t: (-t[0]["block_share"], t[0]["seq"]))
            self.worst = self.worst[:3]
        pic = picture(expected)
        delta = np.abs(bb.astype(np.int16) - window_of(pic).astype(np.int16))
        rt_delta = np.abs(rt.astype(np.int16) - pic.astype(np.int16))
        used = np.any(delta == 1, axis=2)
        self.honesty.append({"seq": f["seq"], "max_channel_delta": int(delta.max()),
                             "bad_pixels": int(np.count_nonzero(np.any(delta > 1, axis=2))),
                             "render_target_bad_pixels": int(np.count_nonzero(np.any(rt_delta > 1, axis=2))),
                             "bound_source_mismatch": bound_id != f["source_fresh_id"],
                             "tolerance_pixels": int(used.sum()), "tolerance_share": float(used.mean())})
        if f["kind"] == "fast":
            self.ratios.append(float((shown != pm).mean()))
            if pm.any() and (~pm).any() and not (shown.any() and (~shown).any()):
                self.both_bad.append(f["seq"])

    def finish(self, out=None):
        paths = []
        if out is not None:
            from PIL import Image
            d = Path(out)
            d.mkdir(parents=True, exist_ok=True)
            for rank, (r, rt, pm) in enumerate(self.worst, 1):
                path = d / f"worst_{rank}_seq{r['seq']}.png"
                Image.fromarray(np.concatenate((rt[::-1, :, :3], pm[::-1, :, :3]), axis=1)).save(path)
                paths.append(str(path))
        shares = [r["block_share"] for r in self.blocks]
        c5 = {"frames": len(self.blocks), "moving_frames": len(self.ratios), "per_present": self.blocks,
              "bad_blocks": [r["seq"] for r in self.blocks if not r["ok"]],
              "distribution": {str(q): float(np.percentile(shares, q)) if shares else None for q in (0, 50, 95, 100)},
              "max_block_share": max(shares, default=0.), "worst": [r for r, _, _ in self.worst], "captures": paths}
        q = {"frames": len(self.ratios), "mean": float(np.mean(self.ratios)) if self.ratios else None,
             "max": max(self.ratios, default=None), "both_judgements_missing": self.both_bad}
        result = {"frames": self.honesty, "condition5": c5, "quality": q,
                  "bad_pixels": sum(r["bad_pixels"] for r in self.honesty),
                  "render_target_bad_pixels": sum(r["render_target_bad_pixels"] for r in self.honesty),
                  "bound_source_mismatch": [r["seq"] for r in self.honesty if r["bound_source_mismatch"]],
                  "tolerance_pixels": sum(r["tolerance_pixels"] for r in self.honesty)}
        if out is not None:
            (Path(out) / "condition5.json").write_text(json.dumps(c5, indent=1), encoding="utf-8")
        return result


def check_replay(log_dir, out=None):
    """Protected window launch; callers must explicitly choose this window check."""
    import launch
    meta, frames, _ = load(log_dir)
    if is_b2_recording(frames, meta):
        fresh, _ = load_fresh(log_dir)
        records = validate_present_records(frames, fresh, meta)
        if any(records.values()):
            raise ValueError(f"B2 present record error: {records}")
    d = Path(out) if out is not None else Path(log_dir) / "hybrid_replay"
    from cpu_guard import guarded_run
    d = guarded_run(lambda directory: launch.run(F / meta["script"], "honesty", directory,
                    replay=Path(log_dir).resolve(), timeout=3600), d)
    return json.loads((d / "honesty.json").read_text(encoding="utf-8"))


def check_present_blocks(log_dir):
    return check_replay(log_dir)["condition5"]



def load_auxiliary(log_dir, fresh, frames):
    d=Path(log_dir)
    margins,precise={},{}
    screen_bits=np.load(d/"fresh_maps.npy",mmap_mode="r",allow_pickle=False)
    screen_rows={ident:n for n,ident in enumerate(fresh)}
    with np.load(d/"fresh_margins.npz",allow_pickle=False) as archive:
        for ident,r in fresh.items():
            if r.get("margin_t_end") is not None:
                shape=r["margin_shape"]
                bits=archive[str(ident)]
                if bits.dtype!=np.uint8 or bits.size!=(int(np.prod(shape))+7)//8:
                    raise ValueError("invalid packed margin archive")
                m=np.unpackbits(bits)[:int(np.prod(shape))].reshape(shape).astype(bool)
                mx,my=r["margin_width"]
                screen=np.unpackbits(screen_bits[screen_rows[ident]]).reshape(1920,1080).astype(bool)
                if sha(m)!=r["margin_archive"]["sha256"] or not np.array_equal(m[mx:mx+1920,my:my+1080],screen):
                    raise ValueError("margin archive SHA/screen mismatch")
                margins[ident]=m
    with np.load(d/"precise_maps.npz",allow_pickle=False) as archive:
        for f in frames:
            if f["kind"]!="fast":
                bits=archive[f["precise_id"]]
                if bits.shape!=(259200,) or bits.dtype!=np.uint8:
                    raise ValueError("invalid packed precise archive")
                m=np.unpackbits(bits).reshape(1920,1080).astype(bool)
                if sha(m)!=f["map_sha256"]:
                    raise ValueError("precise archive SHA mismatch")
                precise[f["precise_id"]]=m
    return margins,precise


def _b3_axes(frame, source, transform):
    # Independently transcribed screen-coordinate expression, no production helpers.
    xd,yd,e=map(float,frame["display_params"])
    xs,ys,s=map(float,source)
    ey,sy=e*(9/16),s*(9/16)
    ox=((xd-xs)-e+s)/(2*s/1919); oy=((yd-ys)-ey+sy)/(2*sy/1079)
    x=((xd-xs)-e+s+np.arange(1920,dtype=np.float64)*(2*e/1919))/(2*s/1919)
    y=((yd-ys)-ey+sy+np.arange(1080,dtype=np.float64)*(2*ey/1079))/(2*sy/1079)
    want=(e/s,ox,oy)
    if not same_bits((transform["scale"],*transform["offset"]),want):
        raise ValueError("B3 transform differs")
    return np.floor(x+.5),np.floor(y+.5)


def _b3_counts_classes(frame, fresh, descriptors, precise_params):
    """Exact counts of the first-covering rule. Columns (rows) with identical per-candidate cover/inside bits form
    one class; a pixel's outcome depends only on its (column class, row class), weighted by class sizes."""
    xbits,ybits,parts=[],[],[]
    for c in descriptors:
        ident,part=c["id"],c["part"]
        p=precise_params[ident] if part=="precise" else fresh[ident]["params"]
        x,y=_b3_axes(frame,p,c["transform"])
        mx,my=fresh[ident]["margin_width"] if part=="screen+margin" else (0,0)
        xbits += [(x>=-mx)&(x<1920+mx),(x>=0)&(x<1920)]
        ybits += [(y>=-my)&(y<1080+my),(y>=0)&(y<1080)]
        parts.append(part)
    counts=dict(newest=0,older=0,margin=0,precise=0,edge=0)
    if not parts:
        counts["edge"]=1920*1080
        return counts
    ux,wx=np.unique(np.stack(xbits,axis=1),axis=0,return_counts=True)
    uy,wy=np.unique(np.stack(ybits,axis=1),axis=0,return_counts=True)
    cov=ux[:,None,0::2]&uy[None,:,0::2]
    ins=ux[:,None,1::2]&uy[None,:,1::2]
    weight=wx[:,None].astype(np.int64)*wy[None,:].astype(np.int64)
    anyc=cov.any(axis=2)
    first=cov.argmax(axis=2)
    inside=np.take_along_axis(ins,first[...,None],axis=2)[...,0]
    precise=np.asarray([q=="precise" for q in parts])[first]
    counts["edge"]=int(weight[~anyc].sum())
    hit=anyc
    counts["precise"]=int(weight[hit&precise].sum())
    counts["newest"]=int(weight[hit&~precise&inside&(first==0)].sum())
    counts["older"]=int(weight[hit&~precise&inside&(first>0)].sum())
    counts["margin"]=int(weight[hit&~precise&~inside].sum())
    return counts


def independent_b3(frame, fresh, maps=None, margins=None, precise_maps=None, *, descriptors=None, precise_params=None):
    """Independent per-pixel coverage/composition; structural mode needs no map bytes."""
    descriptors=frame["candidates"] if descriptors is None else descriptors
    if maps is None:  # speed-up: same per-pixel rule on column/row equivalence classes
        return None,_b3_counts_classes(frame,fresh,descriptors,precise_params)
    remaining=np.ones((1920,1080),bool)
    output=np.empty_like(remaining) if maps is not None else None
    counts=dict(newest=0,older=0,margin=0,precise=0,edge=0)
    fallback=None
    for n,c in enumerate(descriptors):
        ident,part=c["id"],c["part"]
        p=precise_params[ident] if part=="precise" else fresh[ident]["params"]
        x,y=_b3_axes(frame,p,c["transform"])
        mx,my=fresh[ident]["margin_width"] if part=="screen+margin" else (0,0)
        inside=((x>=0)&(x<1920))[:,None]&((y>=0)&(y<1080))[None,:]
        cover=((x>=-mx)&(x<1920+mx))[:,None]&((y>=-my)&(y<1080+my))[None,:]
        take=remaining&cover
        if part=="precise":
            counts["precise"]+=int(take.sum())
        else:
            counts["newest" if n==0 else "older"]+=int(np.count_nonzero(take&inside))
            counts["margin"]+=int(np.count_nonzero(take&~inside))
        if output is not None and (take.any() or n==0):
            m=precise_maps[ident] if part=="precise" else margins[ident] if part=="screen+margin" else maps[ident]
            sampled=m[np.clip(x+mx,0,1919+2*mx).astype(np.int64)[:,None],
                      np.clip(y+my,0,1079+2*my).astype(np.int64)[None,:]]
            if n==0:
                fallback=sampled
            output[take]=sampled[take]
        remaining &= ~cover
        if not remaining.any():  # speed-up: later candidates take no pixel; still resolve their keys
            for rest in descriptors[n+1:]:
                _ = precise_params[rest["id"]] if rest["part"]=="precise" else fresh[rest["id"]]["params"]
                if rest["part"]=="screen+margin":
                    _ = fresh[rest["id"]]["margin_width"]
            break
    counts["edge"]=int(remaining.sum())
    if output is not None:
        output[remaining]=fallback[remaining]
    return output,counts


def derived_margin_completion(record):
    """Validate ring geometry/times independently, including incomplete rings.

    Chunk t_complete is observed after the CUDA event following pinned download.
    margin_t_end also includes the CPU assembly; eligibility must include both.
    """
    attempts = record.get("margin_attempts")
    if attempts is not None:
        if (not isinstance(attempts, list) or not attempts
                or any(not isinstance(a, dict) for a in attempts)):
            raise ValueError("invalid margin attempts")
        if any(a.get("status") != "cancelled" or a.get("margin_t_end") is not None
               for a in attempts[:-1]):
            raise ValueError("previous margin attempt not cancelled")
        final = attempts[-1]
        status = final["status"]
        if status not in ("running", "cancelled", "completed"):
            raise ValueError("invalid margin attempt status")
        if (record.get("margin_chunks", []) != final["margin_chunks"]
                or record.get("margin_t_end") != final["margin_t_end"]
                or (status == "completed") != (final["margin_t_end"] is not None)):
            raise ValueError("margin attempt aliases/completion disagree")
        if status == "cancelled":
            return None  # marked cancelled attempts never contribute geometry or times
    end = record.get("margin_t_end")
    chunks = record.get("margin_chunks", [])
    if end is None and not chunks:
        return None
    mx, my = record["margin_width"]
    if any(type(v) is not int or v < 0 for v in (mx, my)):
        raise ValueError("invalid margin width")
    nx, ny = 1920 + 2*mx, 1080 + 2*my
    if record["margin_shape"] != [nx, ny] or not (mx or my):
        raise ValueError("invalid margin shape")
    rects, completed = [], []
    pending = False
    for chunk in chunks:
        a, b, c, d = chunk["rect"]
        if (any(type(v) is not int for v in (a,b,c,d))
                or not (0 <= a < b <= nx and 0 <= c < d <= ny)
                or (a < mx+1920 and b > mx and c < my+1080 and d > my)):
            raise ValueError("chunk outside ring")
        launch, done = chunk["t_launch"], chunk["t_complete"]
        if not math.isfinite(launch):
            raise ValueError("invalid chunk launch")
        if done is None:
            pending = True
        elif not math.isfinite(done) or launch > done:
            raise ValueError("chunk completion before launch")
        else:
            download = chunk.get("t_download_complete", done)
            if not math.isfinite(download) or download < done:
                raise ValueError("invalid chunk download")
            completed.append(download)
        rects.append((a,b,c,d))
    # Sweep all x boundaries. Sorted y intervals must be disjoint and, for a
    # completed ring, cover exactly the ring intervals at every strip.
    xs = sorted({0,mx,mx+1920,nx} | {v for r in rects for v in r[:2]})
    for left, right in zip(xs, xs[1:]):
        if left == right:
            continue
        intervals = sorted((c,d) for a,b,c,d in rects if a <= left and b >= right)
        merged = []
        for c,d in intervals:
            if merged and c < merged[-1][1]:
                raise ValueError("overlapping margin chunks")
            if merged and c == merged[-1][1]:
                merged[-1] = (merged[-1][0], d)
            else:
                merged.append((c,d))
        expected = [(0,ny)] if left < mx or left >= mx+1920 else [(0,my),(my+1080,ny)]
        expected = [(a,b) for a,b in expected if a < b]
        if end is not None and merged != expected:
            raise ValueError("margin chunks do not tile ring")
    if end is None:
        return None  # partial/cancelled work is never usable
    if pending or not completed or not math.isfinite(end) or end < max(completed):
        raise ValueError("margin_t_end contradicts chunk completion/download")
    return max(end, max(completed))


def b3_record_checks(frames,fresh):
    """Fault m/n gates, independently derive usable parts and SHA first-show lower bounds."""
    margin_bad,sha_bad=[],[]
    completion, chunk_bad = {}, []
    for ident, record in fresh.items():
        try:
            completion[ident] = derived_margin_completion(record)
        except (KeyError,ValueError,TypeError,OverflowError):
            completion[ident] = None
            chunk_bad.append(ident)
    first={}
    precise_params={f["precise_id"]:f["display_params"] for f in frames if f["kind"]!="fast"}
    for f in frames:
        if f["kind"]!="fast":
            continue
        first.setdefault(f["source_fresh_id"],f["t_present"])
        try:
            want=[]
            for c in f["candidates"]:
                expected=dict(c)
                if c["part"]!="precise":
                    r=fresh[c["id"]]
                    end=completion[c["id"]]
                    usable=end is not None and end<f["t_decide"]
                    expected["part"]="screen+margin" if usable else "screen"
                    if expected["part"]!=c["part"]:
                        margin_bad.append(f["seq"])
                want.append(expected)
            _,counts=independent_b3(f,fresh,descriptors=want,precise_params=precise_params)
            if counts!=f["pixel_counts"]:
                margin_bad.append(f["seq"])
        except (KeyError,ValueError,TypeError,OverflowError,ZeroDivisionError):
            margin_bad.append(f["seq"])
    last=max((f["t_present"] for f in frames),default=0.)
    records=[(r,first.get(i,last),i) for i,r in fresh.items()]
    records.extend((r["margin_archive"],first.get(i,last),"margin_"+str(i)) for i,r in fresh.items() if "margin_archive" in r)
    records.extend((f,f["t_present"],f.get("precise_id",f["seq"])) for f in frames if f["kind"]!="fast")
    for r,lower,ident in records:
        try:
            for prefix in ("sha","pack"):
                start,end=r[prefix+"_t_start"],r[prefix+"_t_end"]
                thread=r[prefix+"_thread"]
                if not (math.isfinite(start) and math.isfinite(end) and lower<=start<=end
                        and thread["name"]!="MainThread" and isinstance(thread["ident"],int)):
                    raise ValueError("SHA/pack ordering or worker")
        except (KeyError,ValueError,TypeError):
            sha_bad.append(ident)
    return {"margin_ignored":sorted(set(margin_bad)),"margin_records":chunk_bad,
            "sha_on_path":list(dict.fromkeys(sha_bad))}


def validate_b3(frames,fresh):
    bad_transform,bad_bound,bad_order=[],[],[]
    shown={}
    last_precise=None
    precise_params={f["precise_id"]:f["display_params"] for f in frames if f["kind"]!="fast"}
    previous=-math.inf
    for index,f in enumerate(frames):
        if f["seq"]!=index or f["t_present"]<previous:
            bad_order.append(f["seq"])
        previous=f["t_present"]
        if f.get("bound_source_fresh_id","missing")!=f["source_fresh_id"]:
            bad_bound.append(f["seq"])
        if f["kind"]!="fast":
            shown={}
            last_precise=f["precise_id"]
            try:
                tr={"id":last_precise,"scale":1.0,"offset":[0.0,0.0]}
                want=[{"id":last_precise,"part":"precise","transform":tr}]
                if (not same_bits((f["scale"],*f["offset"]),(1.,0.,0.))
                    or f["candidates"]!=want or f["candidate_ids"]!=[last_precise]
                    or f["candidate_transforms"]!=[tr]
                    or f["pixel_counts"]!=dict(newest=0,older=0,margin=0,precise=2073600,edge=0)):
                    raise ValueError("precise candidate record")
                if f["bound_candidate_ids"]!=[last_precise]:
                    bad_bound.append(f["seq"])
            except (KeyError,ValueError,TypeError):
                bad_transform.append(f["seq"])
            continue
        shown.setdefault(f["source_fresh_id"],f["seq"])
        ids=sorted(shown,key=lambda i:(shown[i],i),reverse=True)[:64]
        if last_precise is not None:
            ids.append(last_precise)
        try:
            cs=f["candidates"]
            if ([c["id"] for c in cs]!=ids or f["candidate_ids"]!=ids
                or [c["transform"] for c in cs]!=f["candidate_transforms"]):
                raise ValueError("candidate order/transform aliases")
            if f["bound_candidate_ids"]!=ids:
                bad_bound.append(f["seq"])
            for c in cs:
                if c["id"]==last_precise and c["part"]!="precise":
                    raise ValueError("precise part")
                if c["part"] not in ("screen","screen+margin","precise"):
                    raise ValueError("part")
            _,counts=independent_b3(f,fresh,precise_params=precise_params)
            if counts!=f["pixel_counts"] or sum(counts.values())!=2073600:
                raise ValueError("coverage counts")
            tr=cs[0]["transform"]
            if not same_bits((f["scale"],*f["offset"]),(tr["scale"],*tr["offset"])):
                raise ValueError("newest transform alias")
            if f["present_kind"]=="fresh" and not same_bits(f["display_params"],fresh[f["source_fresh_id"]]["params"]):
                raise ValueError("fresh params")
        except (KeyError,ValueError,TypeError,OverflowError,ZeroDivisionError):
            bad_transform.append(f["seq"])
    result={"bad_transform":bad_transform,"bad_bound_source":bad_bound,"bad_order":bad_order}
    result.update(b3_record_checks(frames,fresh))
    return result
