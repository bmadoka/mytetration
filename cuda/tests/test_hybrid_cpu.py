"""Hybrid contract controls: CPU only, no CUDA allocation and no window."""
import copy
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

F = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(F / "viewer"), str(F / "tests")]
import hybrid
import viewer_check as vc


def finished(durations):
    return [{"t_start": 0.0, "t_end": d} for d in durations]


@pytest.mark.parametrize("old,times,want,m", [
    ("fresh-only", [], "fresh-only", None), ("warped", [], "fresh-only", None),
    ("fresh-only", [.033], "fresh-only", .033), ("fresh-only", [.033001], "warped", .033001),
    ("warped", [.028], "fresh-only", .028), ("warped", [.028001], "warped", .028001),
    ("fresh-only", [.2] * 20 + [.01] * 8, "fresh-only", .01),
    ("fresh-only", [.01, .09], "warped", (.01 + .09) / 2),
])
def test_mode_boundaries_and_last_eight(old, times, want, m):
    assert hybrid.decide_mode(old, finished(times)) == (want, m)


def scalar_axes(display, source):
    xd, yd, ed = display
    xs, ys, es = source
    ey, sy = ed * (9 / 16), es * (9 / 16)
    ix = [min(1919, max(0, math.floor((((xd-xs)-ed+es+px*(2*ed/1919))/(2*es/1919))+.5)))
          for px in range(1920)]
    iy = [min(1079, max(0, math.floor((((yd-ys)-ey+sy+py*(2*ey/1079))/(2*sy/1079))+.5)))
          for py in range(1080)]
    return ix, iy


@pytest.mark.parametrize("source,display", [
    ((0., 0., 50.), (0., 0., 50.)),
    ((-.465, .918, 4e-9), (-.465 + 1e-10, .918 - 1e-9, 2e-9)),
    ((0., 0., 1.), (100., -100., 1.)),
    ((1., -2., .1), (1.1, -1.9, 1.)),
    ((0., 0., 1919.), (1., 0., 1919.)),  # exact half-index tie rounds upward
])
def test_warp_against_scalar_and_independent_cpu(source, display):
    ix, iy, scale, offset = hybrid.warp_indices(display, source)
    sx, sy = scalar_axes(display, source)
    assert ix.dtype == iy.dtype == np.int32
    assert np.array_equal(ix, sx) and np.array_equal(iy, sy)
    m = np.random.default_rng(17).integers(0, 2, (1920, 1080)).astype(bool)
    assert np.array_equal(vc.independent_warp(m, display, source), m[np.ix_(ix, iy)])
    assert scale == display[2] / source[2] and len(offset) == 2
    if source == display:
        assert np.array_equal(ix, np.arange(1920)) and np.array_equal(iy, np.arange(1080))


def records():
    fresh, frames = [], []
    for i in range(6):
        p = (float(i), 0., 50.)  # each first-shown map is an actual B2 replacement
        hx = [v.hex() for v in p]
        start = 10. + .2 * i
        r = {"id": i, "params": p, "params_hex": hx, "display_params_hex": hx,
             "t_decide": start, "k": i, "t_start": start, "t_end": start+.04,
             "precision": "f64", "async": False, "mode": "warped"}
        fresh.append(r)
        frames.append({"seq": i, "kind": "fast", "present_kind": "warped", "params": p,
                       "display_params": p, "params_hex": hx, "display_params_hex": hx,
                       "t_decide": start, "k": i, "t_mode_decide": start+.041,
                       "t_present": start+.05, "source_fresh_id": i, "bound_source_fresh_id": i,
                       "scale": 1., "offset": [0., 0.], "mode": "warped",
                       "m": (start+.04)-start})
    meta = {"hybrid_version": 1, "input_t": {"0": 10., "5": 11.1}, "monitor_refresh_hz": 75.}
    return meta, frames, fresh


def test_condition2_covers_first_middle_and_tail_and_start_bits():
    meta, frames, fresh = records()
    r = vc.replacement_check(meta, frames, fresh)
    assert r["ok"] and r["max_gap_s"] <= .25
    frozen = [dict(f, source_fresh_id=0) for f in frames]
    assert vc.replacement_check(meta, frozen, fresh)["max_gap_s"] > 1.
    for keep in (frames[2:], frames[:-2], frames[:1] + frames[4:]):
        assert not vc.replacement_check(meta, keep, fresh)["ok"]
    corrupt = copy.deepcopy(fresh)
    corrupt[2]["params"] = (np.nextafter(0., 1.), 0., 50.)
    assert vc.replacement_check(meta, frames, corrupt)["bad_start_params"] == [2]
    corrupt = copy.deepcopy(fresh)
    corrupt[2]["t_decide"] = corrupt[1]["t_decide"]
    assert vc.replacement_check(meta, frames, corrupt)["bad_start_params"] == [2]


def test_condition4_reconstructs_available_records_not_future_or_logged_mode():
    _, frames, fresh = records()
    assert vc.replay_modes(frames, fresh)["bad"] == []
    broken = copy.deepcopy(frames)
    broken[1]["mode"] = "fresh-only"
    assert vc.replay_modes(broken, fresh)["bad"] == [1]
    broken = copy.deepcopy(frames)
    broken[3]["source_fresh_id"] = 2
    assert vc.replay_modes(broken, fresh)["bad"] == [3]
    # A long future computation must not influence an earlier mode decision.
    future = dict(fresh[-1], id=100, t_start=100., t_end=200.)
    assert vc.replay_modes(frames, fresh + [future]) == vc.replay_modes(frames, fresh)
    broken = copy.deepcopy(frames)
    broken[0]["m"] += 1e-10
    assert vc.replay_modes(broken, fresh)["bad"] == [0]


def test_condition1_hybrid_off_cannot_hide_as_exception_and_low_refresh():
    meta, frames, fresh = records()
    modes = vc.replay_modes(frames, fresh)
    slow = vc.present_speed(meta, frames, modes)
    assert not slow["ok"] and slow["exception"]["count"] == 0
    broken = [dict(f, mode="fresh-only") for f in frames]
    assert not vc.present_speed(meta, broken, vc.replay_modes(broken, fresh))["ok"]
    # Fixed 50Hz display uses 47.5 fps and 21ms, rather than silently requiring 60Hz.
    meta["monitor_refresh_hz"] = 50.
    frames = [dict(frames[0], seq=i, t_present=10.+.02*(i+1)) for i in range(50)]
    modes = {"per_present": [{"seq": i, "mode": "warped", "m": .2} for i in range(50)]}
    r = vc.present_speed(meta, frames, modes)
    assert r["ok"] and r["below_60_hz"] and r["fps_limit"] == 47.5 and r["p95_limit_ms"] == 21.
    # All valid fresh-only 25ms intervals are reported as exceptions.
    frames = [dict(frames[0], seq=i, t_present=10.+.025*(i+1)) for i in range(40)]
    modes = {"per_present": [{"seq": i, "mode": "fresh-only", "m": .025} for i in range(40)]}
    r = vc.present_speed(meta, frames, modes)
    assert r["ok"] and r["exception"]["count"] == 40 and r["exception"]["length_s"] == 1.


def test_condition5_checks_actual_full_target_and_uses_unchanged_detector():
    _, frames, fresh = records()
    pm = np.random.default_rng(101).integers(0, 2, (1920, 1080)).astype(bool)
    from deepzoom_blocks import synthetic_block
    checks = vc.ReplayChecks()
    for f in frames:
        actual = synthetic_block(pm) if f["seq"] == 3 else pm
        checks.add(f, vc.picture(actual), vc.window_of(vc.picture(pm)), pm, pm, f["source_fresh_id"])
    r = checks.finish()
    assert r["condition5"]["frames"] == len(frames) and len(r["condition5"]["worst"]) == 3
    assert r["condition5"]["bad_blocks"] == [3] and r["condition5"]["distribution"]["100"] > .02
    assert r["bad_pixels"] == 0 and r["render_target_bad_pixels"] > 0
    # Corruption solely at odd RT pixels is invisible in the reduced window.
    rt = vc.picture(pm)
    rt[1::2, 1::2, :3] = 255 - rt[1::2, 1::2, :3]
    checks = vc.ReplayChecks()
    checks.add(frames[0], rt, vc.window_of(vc.picture(pm)), pm, pm, 0)
    r = checks.finish()
    assert r["bad_pixels"] == 0 and r["render_target_bad_pixels"] > 0
    assert r["quality"]["max"] > 0


def test_packed_archive_preserves_completion_order_ids(tmp_path):
    pm = np.random.default_rng(17).integers(0, 2, (1920, 1080)).astype(bool)
    (tmp_path / "fresh.jsonl").write_text('\n'.join(json.dumps({"id": i}) for i in (2, 0)))
    np.save(tmp_path / "fresh_maps.npy", np.stack([np.packbits(pm), np.packbits(~pm)]))
    records, maps = vc.load_fresh(tmp_path)
    assert set(records) == {0, 2} and np.array_equal(maps[2], pm) and np.array_equal(maps[0], ~pm)


def test_all_fault_anchors_unique_and_payloads_compile():
    import fault_injection_viewer as fi
    tree = ast.parse(Path(fi.__file__).read_text(encoding="utf-8"))
    table = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "FAULTS" for t in n.targets))
    keys = [k.value for k in table.keys]
    assert len(keys) == len(set(keys))
    assert {"interactive_wait", "callback_origin"} <= set(keys)
    for key, (_, path, old, new, _, signal) in fi.FAULTS.items():
        source = path.read_text(encoding="utf-8")
        assert source.count(old) == 1, key
        compile(source.replace(old, new), str(path), "exec")
        assert signal.startswith("SIG:")
    assert {"a", "b", "c", "d", "e", "f", "g", "g2", "h", "i", "j", "k"} <= set(fi.FAULTS)


def test_real_loop_async_completion_and_precise_drain_without_window(monkeypatch):
    import app
    from types import SimpleNamespace
    clock = [0.0]
    monkeypatch.setattr(app.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(app.time, "sleep", lambda t: clock.__setitem__(0, clock[0] + t))
    calls = {"fast": 0, "precise": 0, "precision": []}
    jobs, precise_times = [], []
    p = (0., 0., 50.)
    m = np.zeros((2, 2), dtype=bool)

    def fast_map(*p):
        calls["fast"] += 1
        calls["precision"].append("f64")
        clock[0] += .04
        return m

    def precise_map(*p):
        assert all(j.poll() is not None for j in jobs)
        precise_times.append(clock[0])
        calls["precise"] += 1
        return m

    class Job:
        def __init__(self, *p):
            self.t_start = clock[0]
            self.precision = "f64"
            calls["fast"] += 1
            calls["precision"].append("f64")
            jobs.append(self)

        def poll(self):
            return m if clock[0] >= self.t_start + .06 else None

    v = app.Viewer.__new__(app.Viewer)
    v.__dict__.update(state=p, prev_state=p, display=p, decision_t=None, applied=-1, drawn=-1,
                      last_params=None, last_kind=None, first_new_input_t=None, last_input_t=None,
                      log_dir=None, input_t={}, input_log=[], frames=[], fresh=[], fresh_bits=[], next_fresh_id=0,
                      job=None, job_record=None, source=None, source_map=None, source_texture=False,
                      hybrid_mode="fresh-only", skipped_same_params=0, pool_max=0, recorded=[],
                      mode="timed", readback="record", meta={"t_playback_start": 0., "script_duration": 1., "mode": "timed"})
    v.fast = SimpleNamespace(fast_map=fast_map, precise_map=precise_map, FreshJob=Job,
                             calls=calls, USE_CYCLE_EXIT=False, pool_bytes=lambda: 0)

    def present():
        clock[0] += .013
        return 0

    v.disp = SimpleNamespace(win=None, upload_source=lambda m: None, draw=lambda m: None, present=present,
                             draw_warped=lambda p, s: (p[2] / s[2], (0., 0.)),
                             read_back_buffer=lambda: np.zeros((2, 2, 3), np.uint8))
    v.glfw = SimpleNamespace(window_should_close=lambda win: False, poll_events=lambda: None)
    v.frame("initial", 0.)
    events = [{"t": t, "type": "wheel", "steps": 1 / 6} for t in (.014, .02, .04, .1, .15, .2)]
    v._loop(events, None, None, 0., len(events))
    assert v.job is None and len(jobs) > 1 and v.frames[-1]["kind"] == "precise"
    assert len(v.recorded) == len(v.frames)
    assert len(v.fresh) == calls["fast"] and all(r["t_end"] >= r["t_start"] for r in v.fresh)
    assert not vc.replay_modes(v.frames, v.fresh)["bad"]
    meta = dict(v.meta, input_t={str(k): t for k, t in v.input_t.items()})
    assert not vc.expected_displays(meta, v.frames, {"start": p, "events": events})[1]
    assert vc.same_bits(v.display, vc.expected_params({"start": p, "events": events}, 5))
    # Once the latest decided view is settled with no new input, no further job is issued.
    n = len(jobs)
    v._start_fresh(v.state, clock[0])
    assert len(jobs) == n


def test_early_save_drains_job_and_no_log_does_not_retain_bitmaps(tmp_path):
    import app
    from types import SimpleNamespace
    m = np.zeros((1920, 1080), dtype=bool)
    ready = [False]
    v = app.Viewer.__new__(app.Viewer)
    v.__dict__.update(log_dir=tmp_path, fresh=[], fresh_bits=[], source=None, frames=[],
                      recorded=[], media=[], meta={}, input_log=[], source_map=None, source_texture=False,
                      job_record={"id": 0, "params": [0., 0., 50.]})
    v.disp = SimpleNamespace(upload_source=lambda m: None)
    v.job = SimpleNamespace(event=SimpleNamespace(synchronize=lambda: ready.__setitem__(0, True)),
                            poll=lambda: m if ready[0] else None)
    v.save()
    fresh, maps = vc.load_fresh(tmp_path)
    assert v.job is None and set(fresh) == {0} and np.array_equal(maps[0], m)
    assert fresh[0]["sha256"] == vc.sha(m)
    v.log_dir = None
    v.fresh_bits = []
    v._complete_fresh(m, {"id": 1})
    assert v.fresh_bits == []


def test_video_overlay_and_final_precise_coverage_without_encoding():
    import make_realtime_video as video
    frames = [{"kind": "initial", "t_present": 0.},
              {"kind": "fast", "present_kind": "fresh", "precision": "f32", "t_present": .0133333},
              {"kind": "fast", "present_kind": "warped", "precision": "f64", "t_present": 10.1},
              {"kind": "precise", "t_present": 10.2}]
    assert video.frame_label(frames[1]) == "\uc0c8 \u00b7 f32"
    assert video.frame_label(frames[2]) == "\uc62e\uae40 \u00b7 f64"
    assert video.frame_label(frames[3]) == "\uc815\ubc00 \u00b7 double"
    meta = {"monitor_refresh_hz": 75., "t_end": 10.2001}
    times, indices, fps, start, end, hold = video.recording_timeline(meta, frames)
    assert len(indices) == len(times) and indices[-1] == 3 and 2 in indices
    assert fps == 75. and start == 0. and end > meta["t_end"] and hold > 0.
    meta["t_end"] = 12.
    assert video.recording_timeline(meta, frames)[-1] == 0.


def test_hybrid_measure_verdict_requires_each_condition_and_keeps_skips_visible():
    import measure
    r = {"scene": "DZb", "param_mismatch": [], "started_before_input": [], "calls_equal_counted": True,
         "sha_rechecked": 3, "fresh_count": 3, "counted": 10, "sha_recheck_failed": [],
         "last_frame_is_precise_of_last_input": True,
         "precise_checks": [{"seq": 11, "params_bit_equal": True, "started_after_last_input": True,
                             "map_equal": True, "render_target_equal": True}],
         "transitions_s": [1.], "gpu_use_mib": 300, "foreground_is_viewer_after_show": False,
         "events_delivered": 6, "events": 6, "present_failed": [], "window_ok": True,
         "fps": 75., "frame_time_p95_ms": 14.,
         "hybrid": {"condition1": {"ok": True}, "condition2": {"ok": True, "bad_start_params": []}, "condition4": {"bad": [], "warped": 9},
                    "records": {"bad_transform": [], "bad_bound_source": [], "bad_order": []},
                     "duplicate_jobs": {"count": 0, "jobs": []}},
         "present_blocks": {"frames": 10, "moving_frames": 10, "bad_blocks": []},
         "honesty": {"expected_frames": 10, "frames": [{}] * 10, "bad_pixels": 0,
                     "render_target_bad_pixels": 0, "bound_source_mismatch": []},
         "blocks": {"frames": 10, "bad_coordinates": [], "bad_blocks": [], "sha_mismatch": [], "precision_mismatch": []},
         "quality": {"frames": 10, "mean": .01, "max": .02, "both_judgements_missing": []}}
    assert measure.verdict(r, True, "scroll") is True
    assert measure.verdict(dict(r, analysis_skipped=True), True, "scroll") is None
    assert measure.verdict(dict(r, sha_rechecked=2), True, "scroll") is False
    assert measure.verdict(dict(r, present_blocks={"frames": 9, "bad_blocks": []}), True, "scroll") is False
    assert measure.verdict(dict(r, present_blocks={"frames": 10, "bad_blocks": [1]}), True, "scroll") is False
    for cond, mutation in (("condition1", {"ok": False}), ("condition2", {"ok": False, "bad_start_params": []}),
                           ("condition4", {"bad": [1], "warped": 9})):
        bad = copy.deepcopy(r)
        bad["hybrid"][cond] = mutation
        assert measure.verdict(bad, True, "scroll") is False
    for field, value in (("bad_pixels", 1), ("render_target_bad_pixels", 1),
                         ("bound_source_mismatch", [1]), ("frames", [])):
        bad = copy.deepcopy(r)
        bad["honesty"][field] = value
        assert measure.verdict(bad, True, "scroll") is False
    assert measure.verdict(r, False, "scroll") is None
    for field in ("bad_transform", "bad_bound_source", "bad_order", "margin_records",
                  "margin_ignored", "sha_on_path", "candidates", "contamination", "future_b3_error"):
        bad = copy.deepcopy(r)
        bad["hybrid"]["records"][field] = [123]  # unused/cancelled ring error
        for judged in (False, True):
            assert measure.verdict(bad, judged, "scroll") is False
            assert measure.verdict(dict(bad, analysis_skipped=True), judged, "scroll") is False
    bad = copy.deepcopy(r)
    del bad["hybrid"]["duplicate_jobs"]
    for judged in (False, True):
        assert measure.verdict(bad, judged, "scroll") is False
        assert measure.verdict(dict(bad, analysis_skipped=True), judged, "scroll") is False
    bad["hybrid"]["duplicate_jobs"] = {}
    assert measure.verdict(bad, False, "scroll") is False
    # Shallow manipulation must retain its zero-warp proof even at high fps.
    assert measure.verdict(dict(r, scene="D1"), True, "manip") is False


def test_synchronous_mode_uses_map_completion_before_slow_hash(monkeypatch):
    import app
    from types import SimpleNamespace
    clock, hashes = [0.], []
    monkeypatch.setattr(app.time, "perf_counter", lambda: clock[0])
    real_sha = app.hashlib.sha256

    def slow_sha(data):
        clock[0] += .002
        hashes.append(clock[0])
        return real_sha(data)

    monkeypatch.setattr(app.hashlib, "sha256", slow_sha)
    calls = {"fast": 0, "precision": []}

    def fmap(*p):
        clock[0] += .032
        calls["fast"] += 1
        calls["precision"].append("f32")
        return np.zeros((2, 2), dtype=bool)

    p = (0., 0., 50.)
    v = app.Viewer.__new__(app.Viewer)
    v.__dict__.update(state=p, display=p, decision_t=0., applied=0, drawn=-1, last_params=None,
                      first_new_input_t=0., last_input_t=0., job=None, job_record=None,
                      hybrid_mode="fresh-only", frames=[], fresh=[], fresh_bits=[], next_fresh_id=0,
                      log_dir=None, source=None, source_map=None, source_texture=False,
                      skipped_same_params=0, pool_max=0, mode="timed", readback="none")
    v.fast = SimpleNamespace(fast_map=fmap, precise_map=None, calls=calls, USE_CYCLE_EXIT=False,
                             pool_bytes=lambda: 0)
    v.disp = SimpleNamespace(draw=lambda m: None, present=lambda: 0)
    v.frame("fast", 0.)
    assert v.fresh[0]["t_end"] == v.frames[0]["t_compute_end"] == .032
    v._finish_archive()
    assert clock[0] == .034 and len(hashes) == 1
    assert v.fresh[0]["sha_t_start"] >= v.frames[0]["t_present"]
    assert v.frames[0]["mode"] == "fresh-only" and v.frames[0]["m"] == .032


@pytest.mark.parametrize("scene", ["D1", "Z1", "DZa", "DZb"])
def test_hybrid_measure_requires_start_params_in_all_scenes(scene):
    import measure
    # Exercise the independent mandatory record gate without GPU/window work.
    meta, frames, fresh = records()
    h = vc.check_hybrid_records(meta, frames, fresh)
    h["condition2"]["bad_start_params"] = [1]
    r = {"scene": scene, "hybrid": h}
    from unittest.mock import patch
    with patch.object(measure, "integrity", return_value=True):
        assert measure.verdict(r, False) is False


def test_every_transform_and_bound_source_checked_independently():
    _, frames, fresh = records()
    fresh = {r["id"]: r for r in fresh}
    assert not any(vc.validate_present_records(frames, fresh).values())
    for field, value in (("scale", 999), ("offset", [999, 999]), ("offset", [0., -0.]),
                         ("offset", [0.]), ("scale", float("nan"))):
        bad = copy.deepcopy(frames)
        bad[2][field] = value
        assert vc.validate_present_records(bad, fresh)["bad_transform"] == [2]
    bad = copy.deepcopy(frames)
    bad[1]["bound_source_fresh_id"] = 0
    assert vc.validate_present_records(bad, fresh)["bad_bound_source"] == [1]
    del bad[1]["bound_source_fresh_id"]
    assert vc.validate_present_records(bad, fresh)["bad_bound_source"] == [1]
    bad = copy.deepcopy(frames)
    bad[2]["present_kind"] = "fresh"
    bad[2]["display_params"] = [1., 0., 50.]
    assert vc.validate_present_records(bad, fresh)["bad_transform"] == [2]
    assert vc.validate_present_records(list(reversed(frames)), fresh)["bad_order"]
    display = [-.465 + 1e-10, .918 - 1e-9, 2e-9]
    source = [-.465, .918, 4e-9]
    _, _, scale, offset = hybrid.warp_indices(display, source)
    warped = dict(frames[0], display_params=display, scale=scale, offset=offset)
    src = {0: dict(fresh[0], params=source)}
    assert not vc.validate_present_records([warped], src)["bad_transform"]
    # Fresh/initial/precise identity records are checked as well.
    for kind in ("initial", "precise", "fast"):
        f = dict(frames[0], kind=kind, present_kind="fresh" if kind == "fast" else kind,
                 source_fresh_id=0 if kind == "fast" else None,
                 bound_source_fresh_id=0 if kind == "fast" else None, scale=999)
        assert vc.validate_present_records([f], fresh)["bad_transform"] == [0]


def test_precise_cache_is_bit_exact_and_reused_in_one_check():
    calls = []
    pm = np.zeros((1920, 1080), bool)
    cache = vc.PreciseCache(lambda *p: calls.append(p) or pm)
    for p in ((0., 0., 1.), (0., 0., 1.), (-0., 0., 1.), (-0., 0., 1.)):
        assert np.array_equal(cache(*p), pm)
    assert len(calls) == 2


@pytest.mark.parametrize("source,display", [((0., 0., 1919.), (1., 0., 1919.)),
                                          ((-.465, .918, 4e-9), (-.465 + 1e-10, .918, 2e-9))])
def test_recorded_transform_drives_axes_without_changing_half_ties(source, display):
    ix, iy, scale, offset = hybrid.warp_indices(display, source)
    rx, ry, _, _ = hybrid.warp_indices(display, source, (scale, offset))
    assert np.array_equal(ix, rx) and np.array_equal(iy, ry)
    changed, _, _, _ = hybrid.warp_indices(display, source, (scale, (offset[0]+2., offset[1])))
    assert not np.array_equal(changed, rx)


@pytest.mark.parametrize("stale", [False, True])
def test_recorded_replay_shares_update_draw_order_and_detects_stale_upload(monkeypatch, tmp_path, stale):
    import app
    from types import SimpleNamespace
    p = (0., 0., 50.)
    m0 = np.random.default_rng(99).integers(0, 2, (1920, 1080)).astype(bool)
    maps = {0: m0, 1: ~m0}
    fresh = {i: {"id": i, "params": p, "async": i == 1} for i in maps}
    frames = []
    for seq, (kind, present, ident) in enumerate((("initial", "initial", None), ("fast", "fresh", 0),
                                                ("fast", "warped", 0), ("fast", "warped", 1),
                                                ("precise", "precise", None), ("fast", "warped", 1))):
        frames.append({"seq": seq, "kind": kind, "present_kind": present, "source_fresh_id": ident,
                       "bound_source_fresh_id": ident, "display_params": p,
                       "scale": 1., "offset": [0., 0.], "t_present": float(seq)})
    monkeypatch.setattr(vc, "load", lambda d: ({"hybrid_version":1}, frames, {}))
    monkeypatch.setattr(vc, "load_fresh", lambda d: (fresh, maps))
    precise_calls = []
    monkeypatch.setattr(vc, "reference_precise", lambda *p: precise_calls.append(p) or m0)
    v = app.Viewer.__new__(app.Viewer)
    v.__dict__.update(log_dir=tmp_path, meta={}, source_texture=False, source=None,
                      source_map=None, bound_source_fresh_id=None)
    events, texture, drawn = [], [None], [None]

    def upload(m):
        texture[0] = m
        events.append(("upload", vc.sha(m)))

    def draw(m):
        upload(m)
        drawn[0] = vc.picture(m)
        events.append(("fresh",))

    def warp(display, source, transform=None):
        assert transform == (1., [0., 0.])
        drawn[0] = vc.picture(vc.independent_warp(texture[0], display, source))
        events.append(("warped",))

    v.disp = SimpleNamespace(win=None, upload_source=upload, draw=draw, draw_warped=warp,
                             read_render_target=lambda: drawn[0], read_back_buffer=lambda: vc.window_of(drawn[0]),
                             present=lambda: events.append(("present",)) or 0)
    v.glfw = SimpleNamespace(poll_events=lambda: None, window_should_close=lambda w: False)
    real_set, real_draw = v._set_source, v._draw_present
    updates, draws = [], []

    def set_source(m, rec, upload):
        updates.append((rec["id"], upload))
        real_set(m, rec, upload)

    def draw_present(rec, m=None, recorded_transform=False):
        draws.append(rec["seq"])
        real_draw(rec, m, recorded_transform)

    v._set_source, v._draw_present = set_source, draw_present
    if stale:
        # This fault in the timed app upload path also affects replay, rather than
        # being hidden by an independent upload in run_honesty.
        def stale_upload():
            v.source_texture = True
        v._upload_source = stale_upload
    v.run_honesty(tmp_path)
    r = json.loads((tmp_path / "honesty.json").read_text())
    assert updates == [(0, False), (1, True)] and draws == list(range(6))
    assert len(precise_calls) == 1
    assert r["condition5"]["frames"] == len(frames) and r["condition5"]["moving_frames"] == 4
    assert r["expected_frames"] == len(r["frames"]) == 6
    assert json.loads((tmp_path / "condition5.json").read_text())["frames"] == 6
    assert json.loads((tmp_path / "condition6.json").read_text())["expected_frames"] == 6
    assert len(r["condition5"]["captures"]) == 3
    from PIL import Image
    assert all(Image.open(path).size == (3840, 1080) for path in r["condition5"]["captures"])
    if stale:
        assert r["bad_pixels"] > 0 and r["render_target_bad_pixels"] > 0 and r["bound_source_mismatch"] == [3, 5]
    else:
        assert r["bad_pixels"] == r["render_target_bad_pixels"] == 0 and r["bound_source_mismatch"] == []
    frames[3]["scale"] = 999
    events.clear()
    v.run_honesty(tmp_path)
    rejected = json.loads((tmp_path / "condition6.json").read_text())
    assert rejected["records"]["bad_transform"] == [3]
    assert rejected["expected_frames"] == 6 and rejected["frames"] == [] and events == []


def test_timed_draw_samples_bound_source_independently_of_intended_source():
    import app
    from types import SimpleNamespace
    v = app.Viewer.__new__(app.Viewer)
    v.__dict__.update(source_texture=True, bound_source_fresh_id=0, source={"id": 1, "params": (0., 0., 50.)})
    v.disp = SimpleNamespace(draw_warped=lambda p, s: (1., [0., 0.]))
    f = {"kind": "fast", "present_kind": "warped", "display_params": (0., 0., 50.), "source_fresh_id": 1}
    v._draw_present(f)
    assert f["bound_source_fresh_id"] == 0 and f["source_fresh_id"] == 1


def test_measure_cli_defaults_and_partial_analysis_controls():
    import measure
    a = measure.arguments([])
    assert a.runs == 3 and a.scenes == list(measure.SCENES) and a.kinds == list(measure.SCRIPT_KINDS)
    assert not a.no_precise and a.blocks_runs is None
    a = measure.arguments(["--scenes", "DZb", "--kinds", "scroll", "--runs", "2", "--no-precise", "--blocks-runs", "0"])
    assert a.scenes == ["DZb"] and a.kinds == ["scroll"] and a.runs == 2 and a.no_precise and a.blocks_runs == 0
    for args in (["--runs", "0"], ["--blocks-runs", "-1"]):
        with pytest.raises(SystemExit):
            measure.arguments(args)


@pytest.mark.parametrize("no_precise", [False, True])
def test_measure_selects_first_judged_analysis_and_prints_phase_times(monkeypatch, tmp_path, capsys, no_precise):
    import measure
    launches, checked, block_runs, replay_runs = [], [], [], []
    speed = {"ok": True, "monitor_refresh_hz": 75., "below_60_hz": False,
             "normal": {"fps": 75., "p95_ms": 14.},
             "exception": {"count": 0, "length_s": 0., "fps": 0., "p95_ms": 0.}}

    def launch(script, mode, directory, **kwargs):
        assert mode == "timed" and kwargs == {"readback": "precise"}
        launches.append(directory.name)
        return directory

    def check(directory, precise_check=True):
        checked.append((directory.name, precise_check))
        return {"hybrid": {"condition1": speed, "condition2": {"max_gap_s": .2, "bad_start_params": []},
                           "condition4": {"bad": [], "warped": 9}}, "transitions_s": [.5]}

    def blocks(directory, check_blocks=True):
        assert check_blocks is False
        block_runs.append(directory.name)
        return {"min_row": 1920, "min_column": 1080}

    def replay(directory):
        replay_runs.append(directory.name)
        return {"quality": {"mean": .01, "max": .02},
                "condition5": {"distribution": {"50": .0, "95": .0, "100": .0}, "per_present": []}}

    monkeypatch.setattr(measure.launch, "run", launch)
    def guarded(launcher,directory):
        directory.mkdir(parents=True)
        (directory/"contamination.json").write_text(json.dumps({"verdict":{"contaminated":False}}))
        return launcher(directory)
    monkeypatch.setattr(measure,"guarded_run",guarded)
    monkeypatch.setattr(vc, "require_memory", lambda: None)
    monkeypatch.setattr(vc, "check_run", check)
    monkeypatch.setattr(vc, "blocks", blocks)
    monkeypatch.setattr(vc, "check_replay", replay)
    monkeypatch.setattr(measure, "verdict", lambda *args: None)
    args = ["--out", str(tmp_path / "measure"), "--scenes", "DZb", "--kinds", "scroll", "--runs", "4", "--blocks-runs", "1"]
    if no_precise:
        args.append("--no-precise")
    with pytest.raises(SystemExit) as e:
        measure.main(args)
    assert e.value.code == 0 and len(launches) == len(checked) == 4
    assert all(precise == (not no_precise) for _, precise in checked)
    assert block_runs == replay_runs == ([] if no_precise else ["DZb_scroll_run2"])
    result = json.loads((tmp_path / "measure" / "summary.json").read_text())
    assert result["DZb_scroll_run3"]["analysis_skipped"] and result["DZb_scroll_run4"]["analysis_skipped"]
    assert bool(result["DZb_scroll_run2"].get("analysis_skipped")) == no_precise
    output = capsys.readouterr().out
    assert output.count("timed DZb_scroll_run") == 8 and output.count("records/fresh SHA/stop") == 4
    assert output.count("elapsed") == (8 if no_precise else 10)


def test_replay_cli_uses_protected_launch_and_writes_results(monkeypatch, tmp_path):
    import launch
    import importlib.util
    spec = importlib.util.spec_from_file_location("hybrid_replay_cli", F / "scripts/hybrid_replay_check.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    result = {"frames": [{"seq": 0}], "expected_frames": 1, "condition5": {"bad_blocks": [], "moving_frames": 1},
              "bad_pixels": 0, "render_target_bad_pixels": 0, "bound_source_mismatch": [],
              "records": {"bad_transform": [], "bad_bound_source": [], "bad_order": []}}
    monkeypatch.setattr(vc, "load", lambda d: ({"script": "viewer/scripts/DZb_scroll.json", "hybrid_version": 1}, [], {}))

    def guarded_launch(script, mode, directory, **kwargs):
        calls.append((script, mode, kwargs))
        directory.mkdir()
        (directory / "honesty.json").write_text(json.dumps(result))
        (directory / "contamination.json").write_text(json.dumps({"verdict":{"contaminated":False}}))
        return directory

    monkeypatch.setattr(launch, "run", guarded_launch)
    assert cli.main([str(tmp_path / "run"), "--out", str(tmp_path / "out")]) == 0
    assert calls == [(F / "viewer/scripts/DZb_scroll.json", "honesty",
                      {"replay": (tmp_path / "run").resolve(), "timeout": 3600})]


def test_replay_cli_rejects_a_replay_without_moving_presents(monkeypatch, tmp_path):
    """An empty or initial-only replay must not pass (exit 2)."""
    import launch
    import importlib.util
    spec = importlib.util.spec_from_file_location("hybrid_replay_cli2", F / "scripts/hybrid_replay_check.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(vc, "load", lambda d: ({"script": "viewer/scripts/DZb_scroll.json", "hybrid_version": 1}, [], {}))
    for frames, expected, moving in (([], 0, 0), ([{"seq": 0}], 1, 0)):
        result = {"frames": frames, "expected_frames": expected, "condition5": {"bad_blocks": [], "moving_frames": moving},
                  "bad_pixels": 0, "render_target_bad_pixels": 0, "bound_source_mismatch": [],
                  "records": {"bad_transform": [], "bad_bound_source": [], "bad_order": []}}
        out = tmp_path / f"out_{expected}"

        def guarded_launch(script, mode, directory, result=result, **kwargs):
            directory.mkdir()
            (directory / "honesty.json").write_text(json.dumps(result))
            (directory / "contamination.json").write_text(json.dumps({"verdict":{"contaminated":False}}))
            return directory
        monkeypatch.setattr(launch, "run", guarded_launch)
        assert cli.main([str(tmp_path / "run"), "--out", str(out)]) == 2


def test_duplicate_jobs_uses_computing_and_latest_finished_not_all_history():
    def job(i, p, start, end, asynchronous=True):
        return dict(id=i, params=p, t_start=start, t_end=end, mode="warped", **{"async": asynchronous})
    def presents(rows):
        return [{"seq": i, "kind": "fast", "params": r["params"],
                 "t_compute_start": r["t_start"],
                 "t_compute_end": r["t_start"] if r["async"] else r["t_end"],
                 "t_decide": r["t_start"], "mode": "warped"} for i, r in enumerate(rows)]
    a, b = (0., 0., 1.), (1., 0., 1.)
    rows = [job(0, a, 0, 1, False), job(1, a, 1.1, 2), job(2, b, 2.1, 3), job(3, a, 3.1, 4)]
    assert vc.duplicate_job_check(presents(rows), rows)["count"] == 1  # A->B->A allowed after B completes
    assert hybrid.duplicate_params(a, [], [rows[0]])
    assert not hybrid.duplicate_params(a, [], rows[:3])
    overlap = [job(0, a, 0, 5), job(1, b, .1, .5), job(2, a, 1, 2)]
    assert vc.duplicate_job_check(presents(overlap), overlap)["jobs"][0]["matches"] == [0]
    assert hybrid.duplicate_params(a, [overlap[0]], [overlap[1]])
    assert not hybrid.duplicate_params((-0., 0., 1.), [], [rows[0]])
    signed = [rows[0], job(1, (-0., 0., 1.), 1.1, 2)]
    assert vc.duplicate_job_check(presents(signed), signed)["count"] == 0


def test_replacements_ignore_equal_parameters_and_allow_return_to_a():
    meta, frames, fresh = records()
    for i in (1, 2):
        fresh[i]["params"] = fresh[0]["params"]
    r = vc.replacement_check(meta, frames, fresh)
    assert r["times"][:3] == [10., 10.05, frames[3]["t_present"]]
    assert r["max_gap_s"] > .5
    fresh[2]["params"] = fresh[0]["params"]
    fresh[1]["params"] = (1., 0., 50.)
    assert vc.replacement_check(meta, frames, fresh)["replacements"] == 6


def b2_cpu_viewer(monkeypatch, duration):
    import app
    from types import SimpleNamespace
    clock = [1.]
    monkeypatch.setattr(app.time, "perf_counter", lambda: clock[0])
    v = app.Viewer.__new__(app.Viewer)
    p = (0., 0., 50.)
    v.__dict__.update(state=p, display=p, decision_t=0., applied=0, drawn=-1, last_params=None,
                      first_new_input_t=1., last_input_t=1., job=None, job_record=None,
                      hybrid_mode="fresh-only", frames=[], fresh=[], fresh_bits=[], next_fresh_id=0,
                      log_dir=None, source=None, source_map=None, source_texture=False,
                      shown_maps=[], shown_ids=set(), bound_source_fresh_id=None, hybrid_version=2,
                      skipped_same_params=0, pool_max=0, mode="timed", readback="none")
    m = np.zeros((1920, 1080), bool)
    calls = {"fast": 0, "precision": []}
    def fmap(*p):
        clock[0] += duration
        calls["fast"] += 1
        calls["precision"].append("f64" if duration > .033 else "f32")
        return m.copy()
    class Job:
        def __init__(self, *p):
            self.t_start, self.precision = clock[0], "f64"
        def poll(self):
            return None
    v.fast = SimpleNamespace(fast_map=fmap, FreshJob=Job, calls=calls, USE_CYCLE_EXIT=False, pool_bytes=lambda: 0)
    def multi(display, candidates, transforms=None):
        v.disp.bound_candidate_ids = [r["id"] for r, _ in candidates]
        _, tr, counts = hybrid.candidate_plan(display, [r for r, _ in candidates], transforms)
        return tr, counts
    v.disp = SimpleNamespace(draw=lambda m: None, upload_source=lambda m: None,
                             draw_candidates=multi, bound_candidate_ids=[], present=lambda: 0)
    return v, clock


def test_switching_frame_does_not_start_duplicate_job(monkeypatch):
    v, clock = b2_cpu_viewer(monkeypatch, .04)
    v.frame("fast", clock[0])
    assert v.frames[0]["mode"] == "warped"
    if v.job is not None:  # complete an injected job so the independent record checker sees its lifetime
        clock[0] += .04
        v._complete_fresh(v.source_map, v.job_record)
    duplicates = vc.duplicate_job_check(v.frames, v.fresh)
    assert duplicates["count"] == 0, f"SIG:duplicate_jobs {duplicates}"


def test_warped_starts_replacement_for_changed_parameters(monkeypatch):
    v, clock = b2_cpu_viewer(monkeypatch, .04)
    v.frame("fast", clock[0])
    clock[0] += .02
    v.state = v.display = (1., 0., 50.)
    v.applied += 1
    v.frame("fast", clock[0])
    assert v.job is not None, "SIG:fresh_replacement changed parameters must start a new job"
    assert v.job_record["params"] == [1., 0., 50.]
    assert v.job_record["mode"] == "warped"


def test_fresh_only_recomputes_unchanged_display_on_every_refresh(monkeypatch):
    v, clock = b2_cpu_viewer(monkeypatch, .01)
    v.frame("fast", clock[0])
    clock[0] += .02
    v.frame("fast", clock[0])
    assert v.fast.calls["fast"] == len(v.frames) == len(v.fresh) == 2
    assert v.frames[0]["source_fresh_id"] != v.frames[1]["source_fresh_id"]
    assert all(f["present_kind"] == "fresh" and f["compute_ms"] == pytest.approx(10.) for f in v.frames)
    assert len({f["map_sha256"] for f in v.frames}) == 1
    assert v.skipped_same_params == 0


def test_loop_schedules_unchanged_fresh_only_refreshes_until_stop(monkeypatch):
    from types import SimpleNamespace
    v, clock = b2_cpu_viewer(monkeypatch, .01)
    v.frame("fast", clock[0])
    assert v.applied == v.drawn and v.display == v.state
    v.glfw = SimpleNamespace(poll_events=lambda: None, window_should_close=lambda w: len(v.frames) >= 3)
    v.disp.win = None
    v._loop(None, None, None, 1., 0)
    assert v.fast.calls["fast"] == len(v.frames) == 3
    assert all(f["params"] == list(v.state) for f in v.frames)


@pytest.mark.parametrize("display", [(0., 0., 2.), (1., 0., 1.), (9., -9., 1.),
                                           (0., 0., 1.), (0.25, -0.1, 1.3)])
def test_multisource_priority_counts_and_edge_against_independent_cpu(display):
    sources = [{"id": 2, "params": (0., 0., 1.)}, {"id": 1, "params": (1., 0., 2.)},
               {"id": 0, "params": (-1., .5, 3.)}]
    axes, tr, counts = hybrid.candidate_plan(display, sources)
    maps = {r["id"]: np.random.default_rng(r["id"]).integers(0, 2, (1920,1080)).astype(bool) for r in sources}
    actual = np.empty((1920,1080), bool)
    unfilled = np.ones_like(actual)
    for n, ((ix, iy), r) in enumerate(zip(axes, sources)):
        take = unfilled & ((ix >= 0) & (ix < 1920))[:, None] & ((iy >= 0) & (iy < 1080))[None, :]
        sample = maps[r["id"]][np.ix_(np.clip(ix,0,1919), np.clip(iy,0,1079))]
        actual[take] = sample[take]
        if n == 0:
            edge = sample
        unfilled &= ~take
    actual[unfilled] = edge[unfilled]
    f = {"display_params": display, "candidate_ids": [2,1,0], "candidate_transforms": tr}
    expected, counted = vc.independent_candidates(f, {r["id"]:r for r in sources}, maps)
    assert np.array_equal(actual, expected) and counts == counted
    assert sum(counts.values()) == 1920*1080
    if display == (0.,0.,2.):
        assert counts["older"] > 0 and counts["edge"] == 0


def test_first_shown_history_caps_at_eight_and_replay_records_are_validated(monkeypatch):
    v, clock = b2_cpu_viewer(monkeypatch, .01)
    for i in range(10):
        clock[0] += .02
        v.frame("fast", clock[0])
    assert [r["id"] for _, r, _ in v.shown_maps] == list(range(9,1,-1))
    assert v.frames[-1]["candidate_ids"] == list(range(9,1,-1))
    fresh = {r["id"]: r for r in v.fresh}
    assert not any(vc.validate_present_records(v.frames, fresh).values())
    for field, value in (("candidate_ids", [8,9]), ("pixel_counts", {"edge": 0}),
                         ("candidate_transforms", [])):
        broken = copy.deepcopy(v.frames)
        broken[-1][field] = value
        assert vc.validate_present_records(broken, fresh)["bad_transform"] == [9]
    broken = copy.deepcopy(v.frames)
    broken[-1]["bound_candidate_ids"] = [8]
    assert vc.validate_present_records(broken, fresh)["bad_bound_source"] == [9]


def test_multisource_gl_uploads_signed_tables_and_retains_immutable_layers(monkeypatch):
    import render
    gl = render.GL
    uploads, tables = [], []
    for name in ("glActiveTexture", "glBindTexture", "glBindFramebuffer", "glViewport",
                 "glUseProgram", "glUniform1i", "glUniform1iv", "glUniform2iv", "glUniform4iv", "glDrawArrays"):
        monkeypatch.setattr(gl, name, lambda *args: None)
    monkeypatch.setattr(gl, "glGetUniformLocation", lambda *args: 0)
    monkeypatch.setattr(gl, "glTexSubImage3D", lambda *args: uploads.append((args[4], args[-1].copy())))
    monkeypatch.setattr(gl, "glTexSubImage2D", lambda *args: tables.append(args[-1].copy()))
    d = render.Display.__new__(render.Display)
    d.source_layers, d.sources_tex, d.multi_indices = {}, 10, [11,12]
    d.fbo, d.p_multi = 13, 14
    d._draw_window = lambda: None
    m = np.zeros((1920,1080), bool)
    candidates = [({"id":2, "params":(0.,0.,1.)}, m), ({"id":1, "params":(0.,0.,3.)}, ~m)]
    d.draw_candidates((0.,0.,2.), candidates)
    assert len(uploads) == 2 and uploads[0][1].shape == (1920,1080)
    assert tables[0].shape == (2,1920) and tables[1].shape == (2,1080)
    assert tables[0].dtype == np.int32 and tables[0].min() == -1 and tables[0].max() == 1920
    retained = d.source_layers[(2,"screen")]
    d.draw_candidates((0.,0.,2.), candidates)
    assert len(uploads) == 2
    d.draw_candidates((0.,0.,2.), [({"id":3, "params":(0.,0.,2.)}, ~m), candidates[0]])
    assert len(uploads) == 3 and d.source_layers[(2,"screen")] == retained
    assert set(d.source_layers) == {(2,"screen"),(3,"screen")} and d.bound_candidate_ids == [3,2]


def test_multisource_replay_uses_recorded_candidates_and_rejects_corrupt_records(monkeypatch, tmp_path):
    import app
    from types import SimpleNamespace
    v, clock = b2_cpu_viewer(monkeypatch, .04)
    v.frame("fast", clock[0])
    # First source remains stored while a newly finished source becomes visible.
    clock[0] += .1
    rec = dict(v.fresh[0], id=1, params=(1.,0.,50.), **{"async": True})
    v._complete_fresh(~v.source_map, rec, upload=False)
    v.state, v.display = (1.,0.,50.), (1.,0.,50.)
    v.frame("fast", clock[0])
    # The new frame may queue a job only if params changed; first shown pool must match records.
    frames, fresh = copy.deepcopy(v.frames), {r["id"]:r for r in v.fresh}
    maps = {r["id"]: m for _,r,m in v.shown_maps}
    monkeypatch.setattr(vc, "load", lambda d: ({"hybrid_version":2},frames,{}))
    monkeypatch.setattr(vc, "load_fresh", lambda d: (fresh,maps))
    monkeypatch.setattr(vc, "reference_precise", lambda *p: np.zeros((1920,1080),bool))
    v.log_dir, v.meta, v.job, v.job_record = tmp_path, {}, None, None
    drawn, calls = [None], []
    def multi(display, candidates, transforms=None):
        ids = [r["id"] for r,_ in candidates]
        calls.append(ids)
        v.disp.bound_candidate_ids = ids
        # Mock GL selection from production integer tables (not the checker compositor).
        axes, tr, counts = hybrid.candidate_plan(display, [r for r,_ in candidates], transforms)
        out = np.empty((1920,1080),bool)
        remain = np.ones_like(out)
        for i, ((ix,iy), (_,m)) in enumerate(zip(axes,candidates)):
            sampled = m[np.ix_(np.clip(ix,0,1919),np.clip(iy,0,1079))]
            if i == 0:
                edge = sampled
            cover = ((ix>=0)&(ix<1920))[:,None] & ((iy>=0)&(iy<1080))[None,:]
            out[remain&cover] = sampled[remain&cover]
            remain &= ~cover
        out[remain] = edge[remain]
        drawn[0] = vc.picture(out)
        return tr, counts
    v.disp = SimpleNamespace(win=None, upload_source=lambda m: None, draw_candidates=multi,
                             read_render_target=lambda: drawn[0], read_back_buffer=lambda: vc.window_of(drawn[0]),
                             present=lambda: 0, bound_candidate_ids=[])
    v.glfw = SimpleNamespace(poll_events=lambda: None, window_should_close=lambda w: False)
    v.run_honesty(tmp_path)
    result = json.loads((tmp_path/"honesty.json").read_text())
    assert calls == [[0],[1,0]]
    assert result["bad_pixels"] == result["render_target_bad_pixels"] == 0
    frames[-1]["candidate_ids"] = [0,1]
    calls.clear()
    v.run_honesty(tmp_path)
    result = json.loads((tmp_path/"honesty.json").read_text())
    assert result["records"]["bad_transform"] == [1] and calls == []


@pytest.mark.parametrize("field", vc.CANDIDATE_FIELDS + ("all",))
@pytest.mark.parametrize("index", [0, 1])
def test_b2_candidate_omissions_are_record_errors_before_honesty(monkeypatch, tmp_path, field, index):
    import launch
    v, clock = b2_cpu_viewer(monkeypatch, .01)
    for _ in range(2):
        v.frame("fast", clock[0])
        clock[0] += .02
    fresh = {r["id"]: r for r in v.fresh}
    assert not any(vc.validate_present_records(v.frames, fresh).values())
    fields = vc.CANDIDATE_FIELDS if field == "all" else (field,)
    for key in fields:
        del v.frames[index][key]
    records = vc.validate_present_records(v.frames, fresh)
    assert index in records["bad_transform"] + records["bad_bound_source"]
    monkeypatch.setattr(vc, "load", lambda d: ({"script": "unused", "hybrid_version": 2}, v.frames, {}))
    monkeypatch.setattr(vc, "load_fresh", lambda d: (fresh, {}))
    def no_launch(*args, **kwargs):
        pytest.fail("invalid B2 records reached the window launcher")
    monkeypatch.setattr(launch, "run", no_launch)
    with pytest.raises(ValueError, match="B2 present record error"):
        vc.check_replay(tmp_path)


def test_metadata_marker_requires_candidates_and_unmarked_legacy_remains_readable(monkeypatch):
    v, clock = b2_cpu_viewer(monkeypatch, .01)
    v.frame("fast", clock[0])
    fresh = {r["id"]: r for r in v.fresh}
    for key in vc.CANDIDATE_FIELDS:
        del v.frames[0][key]
    assert not vc.is_b2_recording(v.frames, {"hybrid_version": 1})
    assert not any(vc.validate_present_records(v.frames, fresh, {"hybrid_version": 1}).values())
    assert vc.validate_present_records(v.frames, fresh, {"hybrid_version": 2})["bad_transform"] == [0]


def duplicate_launch_records():
    p = (0., 0., 1.)
    rows = [{"id": 0, "async": False, "mode": "fresh-only", "params": p, "t_start": 0., "t_end": .04},
            {"id": 1, "async": True, "mode": "warped", "params": p, "t_start": .041, "t_end": .08}]
    # The job starts in the switching present, whose decided mode is warped.
    frames = [{"seq": 0, "kind": "initial", "t_decide": -.1, "mode": "fresh-only"},
              {"seq": 1, "kind": "fast", "params": p, "t_compute_start": 0., "t_compute_end": .04,
               "t_decide": 0., "mode": "warped"}]
    return frames, rows


@pytest.mark.parametrize("field", ["async", "mode", "params", "t_start", "t_end"])
def test_duplicate_recount_rejects_missing_job_fields(field):
    frames, rows = duplicate_launch_records()
    assert vc.duplicate_job_check(frames, rows)["count"] == 1
    del rows[1][field]
    with pytest.raises(ValueError, match="job record error"):
        vc.duplicate_job_check(frames, rows)


def test_duplicate_recount_rejects_job_mode_disagreement_and_missing_present():
    frames, rows = duplicate_launch_records()
    assert vc.duplicate_job_check(frames, rows)["count"] == 1
    rows[1]["mode"] = "fresh-only"
    with pytest.raises(ValueError, match="launch mode/present disagreement"):
        vc.duplicate_job_check(frames, rows)
    rows[1]["mode"] = "warped"
    with pytest.raises(ValueError, match="job record error.*synchronous ownership"):
        vc.duplicate_job_check([], rows)
    del frames[-1]["mode"]
    with pytest.raises(ValueError, match="launch mode/present disagreement"):
        vc.duplicate_job_check(frames, rows)
