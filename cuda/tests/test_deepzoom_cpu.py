"""No windows or CUDA work: fixed detector controls and the documented zoom contract."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

F = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(F / "viewer"), str(F / "tests")]
import params
import script
import viewer_check as vc
from deepzoom_blocks import compare, synthetic_block, synthetic_shift


def test_detector_controls():
    precise = np.random.default_rng(20261017).integers(0, 2, (200, 120)).astype(bool)
    assert not compare(synthetic_block(precise), precise)["ok"]
    assert compare(synthetic_shift(precise), precise)["ok"]
    assert compare(precise, precise)["block_share"] == 0


@pytest.mark.parametrize("start", [(0.0, 0.0, 50.0), (-0.4650135453678735, 0.9182841487408027, 2.8840315031266057e-7)])
def test_smooth_step_matches_documented_equation(start):
    target = params.apply_wheel(*start, 1.0)
    display = start
    for dt in (0.5, 0.016, 0.011, 0.033, 0.009, 0.075) + (0.016,) * 16:
        want = vc.contract_smooth(display, target, dt)
        got = params.smooth_step(display, target, dt)
        assert vc.same_bits(got, want)
        display = got
    assert vc.same_bits(display, target)
    assert vc.same_bits(params.smooth_step(target, target, 0.016), target)
    moved = (1.25, -2.5, target[2])
    assert vc.same_bits(params.smooth_step(display, moved, 0.0), moved)


@pytest.mark.parametrize("scene", ["D1", "Z1", "DZa", "DZb"])
def test_deepzoom_scripts_are_fixed(scene):
    root = F / "viewer/scripts"
    spec = json.loads((root / "deepzoom_scenes.json").read_text(encoding="utf-8"))["scenes"][scene]
    for kind, generate in (("manip", script.manipulation), ("scroll", script.scroll)):
        data = json.loads((root / f"{scene}_{kind}.json").read_text(encoding="utf-8"))
        assert data["start"] == spec["start"]
        assert data["events"] == generate(spec["start"])
        times = [0.0] + [e["t"] for e in data["events"]]
        assert max(b - a for a, b in zip(times, times[1:])) <= 0.050
        assert len(data["events"]) / 10 >= 60
        if kind == "scroll":
            assert len(data["events"]) == 600 and data["events"][-1]["t"] == 10.0
            assert all(e["type"] == "wheel" and e["steps"] == 1 / 6 for e in data["events"])
            assert vc.expected_params(data, 599)[2] >= 1e-13
        elif scene in ("D1", "Z1"):
            legacy = json.loads((root / f"{scene}.json").read_text(encoding="utf-8"))
            assert data["events"] == legacy["events"]
    if scene in ("D1", "DZb"):
        data = json.loads((root / f"{scene}_one_notch.json").read_text(encoding="utf-8"))
        assert data["events"] == [{"i": 0, "t": 1.0, "type": "wheel", "steps": 1.0}]
        assert data["duration"] == 2.0


def test_independent_display_replay_detects_jump_and_missing_due_input():
    data = {"start": [0.0, 0.0, 50.0], "events": [{"t": 1.0, "type": "wheel", "steps": 1.0}]}
    target = vc.expected_params(data, 0)
    start = tuple(data["start"])
    shown = vc.contract_smooth(start, target, 1.0)
    def frame(seq, p, t, k, kind, target):
        hx = [v.hex() for v in p]
        return {"seq": seq, "params": p, "display_params": p, "params_hex": hx, "display_params_hex": hx,
                "target_params_hex": [v.hex() for v in target], "t_decide": t, "t_compute_start": t,
                "k": k, "last_input": k, "kind": kind, "frame_type": "new_input",
                "t_first_new_input": 11.0 if k == 0 else None}
    frames = [frame(0, start, 10.0, -1, "initial", start), frame(1, shown, 11.0, 0, "fast", target)]
    meta = {"input_t": {"0": 11.0}, "mode": "timed", "t_playback_start": 10.0}
    assert vc.expected_displays(meta, frames, data)[1] == []
    frames[1] = frame(1, target, 11.0, 0, "fast", target)
    assert vc.expected_displays(meta, frames, data)[1] == [1]
    frames[1] = frame(1, start, 11.0, -1, "fast", start)
    assert vc.expected_displays(meta, frames, data)[1] == [1]


def test_precision_rule_and_coordinate_detector_without_cuda():
    import fast
    from mytet_cuda import variants
    scenes = json.loads((F / "viewer/scripts/deepzoom_scenes.json").read_text(encoding="utf-8"))["scenes"]
    for scene, spec in scenes.items():
        p = spec["start"]
        precision = fast.choose_precision(*p)
        assert precision == ("f32" if scene in ("D1", "Z1") else "f64")
        if precision == "f32":
            x, y, _ = variants.grid("16by9_H", *p, 1920)
            assert np.unique(x.astype(np.float32)).size == 1920
            assert np.unique(y.astype(np.float32)).size == 1080
    assert fast.choose_precision(0.0, 0.0, 0.06324054473526401) == "f64"  # measured near-origin block counterexample
    assert fast.choose_precision(0.0, 0.0, 0.1) == "f32"
    c = np.empty((1920, 1080, 2), dtype=np.float32)
    c[..., 0] = np.arange(1920)[:, None]
    c[..., 1] = np.arange(1080)[None, :]
    assert vc.coordinate_counts(c)["ok"]
    c[1, 7, 0] = c[0, 7, 0]
    assert vc.coordinate_counts(c)["min_row"] == 1919


def test_fault_injection_anchors_are_unique():
    import fault_injection_viewer as fi
    for key, (_, path, old, _, _, _) in fi.FAULTS.items():
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1, key


def test_frame_loop_contract_with_no_window_or_gpu(monkeypatch):
    """Real frame/loop methods, fake clock/maps/presentation; inputs due during a
    computation must be drained before the next decision, including exact ties.
    Also exercises record readback on interpolation and precise anchoring.
    """
    import app
    from types import SimpleNamespace
    clock = [0.0]
    def now():
        clock[0] += 0.0001
        return clock[0]
    monkeypatch.setattr(app.time, "perf_counter", now)
    monkeypatch.setattr(app.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    calls = {"fast": 0, "precise": 0, "precision": []}
    def fmap(*p):
        clock[0] += 0.008
        calls["fast"] += 1
        calls["precision"].append("f32")
        return np.zeros((2, 2), dtype=bool)
    def pmap(*p):
        calls["precise"] += 1
        return np.zeros((2, 2), dtype=bool)
    v = app.Viewer.__new__(app.Viewer)
    v.__dict__.update(state=(0.0, 0.0, 50.0), prev_state=(0.0, 0.0, 50.0), display=(0.0, 0.0, 50.0),
                      decision_t=None, applied=-1, drawn=-1, last_params=None, last_kind=None,
                      first_new_input_t=None, last_input_t=None, input_t={}, input_log=[], frames=[],
                      log_dir=None, fresh=[], fresh_bits=[], next_fresh_id=0, job=None, job_record=None, source=None,
                      source_map=None, source_texture=False, hybrid_mode="fresh-only",
                      skipped_same_params=0, pool_max=0, recorded=[], mode="timed", readback="record",
                      meta={"t_playback_start": 0.0, "script_duration": 1.0, "mode": "timed"})
    v.fast = SimpleNamespace(fast_map=fmap, precise_map=pmap, calls=calls, USE_CYCLE_EXIT=False, pool_bytes=lambda: 0)
    v.disp = SimpleNamespace(win=None, draw=lambda m: None, upload_source=lambda m: None, present=lambda: 0,
                             read_back_buffer=lambda: np.zeros((2, 2, 4), dtype=np.uint8))
    ticks = [0]
    def closed(win):
        ticks[0] += 1
        assert ticks[0] < 2000, "loop did not settle"
        return False
    v.glfw = SimpleNamespace(window_should_close=closed, poll_events=lambda: None)
    v.frame("initial", 0.0)
    events = [{"t": t, "type": "wheel", "steps": 1 / 6} for t in (0.0004, 0.004, 0.006)]
    events.append({"t": 0.8, "type": "wheel", "steps": 1 / 6})
    v._loop(events, None, None, 0.0, len(events))
    meta = dict(v.meta, input_t={str(k): t for k, t in v.input_t.items()})
    data = {"start": [0.0, 0.0, 50.0], "events": events}
    assert vc.expected_displays(meta, v.frames, data)[1] == []
    assert any(f["frame_type"] == "interp" and f["kind"] == "fast" for f in v.frames)
    assert v.frames[-1]["kind"] == "precise" and vc.same_bits(v.display, vc.expected_params(data, 3))
    assert any(a["kind"] == "precise" and b["kind"] == "fast" for a, b in zip(v.frames, v.frames[1:]))
    assert len(v.recorded) == len(v.frames)


@pytest.mark.parametrize("kind,limit", [("manip", 50.0), ("scroll", 33.0)])
def test_measure_requires_blocks_quality_and_fixed_time_limits(kind, limit):
    import measure
    r = {"param_mismatch": [], "started_before_input": [], "calls_equal_counted": True,
         "sha_rechecked": 10, "counted": 10, "sha_recheck_failed": [],
         "last_frame_is_precise_of_last_input": True,
         "precise_checks": [{"seq": 9, "params_bit_equal": True, "started_after_last_input": True,
                             "map_equal": True, "render_target_equal": True}],
         "transitions_s": [2.0], "gpu_use_mib": 300, "foreground_is_viewer_after_show": False,
         "events_delivered": 5, "events": 5, "fps": 30.0, "frame_time_p95_ms": limit,
         "present_failed": [], "window_ok": True,
         "quality": {"frames": 10, "mean": 0.10, "max": 0.15, "both_judgements_missing": []},
         "blocks": {"frames": 10, "bad_coordinates": [], "bad_blocks": [], "sha_mismatch": [],
                    "precision_mismatch": []}}
    assert measure.verdict(r, True, kind) is True
    assert measure.verdict(dict(r, frame_time_p95_ms=limit + 0.001), True, kind) is False
    assert measure.verdict(dict(r, transitions_s=[2.0001]), True, kind) is False
    assert measure.verdict(dict(r, fps=29.999), True, kind) is False
    for flag in ("bad_coordinates", "bad_blocks", "sha_mismatch", "precision_mismatch"):
        assert measure.verdict(dict(r, blocks=dict(r["blocks"], **{flag: [1]})), True, kind) is False
    assert measure.verdict(dict(r, blocks=dict(r["blocks"], frames=9)), True, kind) is False


def test_pt32_native_coordinate_representation():
    # Large common reference plus small deltas must not collapse to f32 c.
    gamma = np.empty((1920, 1080, 2), dtype=np.float32)
    gamma[..., 0] = np.arange(1920, dtype=np.float32)[:, None] * 1e-12
    gamma[..., 1] = np.arange(1080, dtype=np.float32)[None, :] * 1e-12
    c = {"reference": np.array([-0.465, 0.918], dtype=np.float64), "gamma": gamma}
    assert vc.coordinate_counts(c)["ok"]
    gamma[10, 7, 0] = gamma[9, 7, 0]
    assert vc.coordinate_counts(c)["min_row"] == 1919
    c["reference"][0] = np.nan
    assert not vc.coordinate_counts(c)["finite"]
    assert not vc.coordinate_counts(c)["ok"]


def test_pt32_reference_order_is_bench_only():
    sys.path.insert(0, str(F / "bench"))
    import deepzoom_pt32
    import fast
    points = deepzoom_pt32._reference_candidates(1920, 1080, 4)
    assert points[0].tolist() == [960, 540]
    assert len(points) == len({tuple(p) for p in points}) == 81
    radii = np.max(np.abs(points - [960, 540]), axis=1)
    assert np.all(radii[1:] >= radii[:-1])
    assert "pt32" not in fast.SOURCE
    assert not hasattr(fast, "PT32_ENABLED")
    with pytest.raises(ValueError):
        fast.fast_map(0, 0, 50, precision="pt32")


def test_g2_uses_plain_f32_and_forces_double_coordinates():
    """g2: c and log c stay double; only the orbit state z is rounded to float32 every step."""
    import fault_injection_viewer as fi
    _, path, old, new, _, _ = fi.FAULTS["g2"]
    source = path.read_text(encoding="utf-8")
    assert source.count(old) == 1
    compile(source.replace(old, new), str(path), "exec")
    assert 'precision = "f64"' in new
    assert 'dc zn = d_cpow(c, dmk((double)zf.re, (double)zf.im), lc);' in new  # exact c and log c
    assert 'zf = fmk((float)zn.re, (float)zn.im);' in new                       # orbit stored in float32
    assert '(float)c.re' in new and 'f_clog(' not in new and 'cf' not in new      # c itself not rounded for the base
    assert 'pt32_map_rows' not in new


def test_d1_manip_large_extent_regression_without_cuda(monkeypatch):
    import fast
    def no_gpu(*args, **kwargs):
        raise AssertionError("precision selection must not touch CUDA")
    monkeypatch.setattr(fast.gpu, "_check_device", no_gpu)
    data = json.loads((F / "viewer/scripts/D1_manip.json").read_text(encoding="utf-8"))
    bad = {62, 65, 66, 93, 222, 225, 226, 253}
    state = tuple(data["start"])
    assert fast.choose_precision(*state) == "f32"
    zoom_count = 0
    for k, ev in enumerate(data["events"]):
        state = params.apply_event(state, ev)
        if k in bad:
            assert fast.choose_precision(*state) == "f64", k
        if ev["type"] == "wheel":
            assert fast.choose_precision(*state) == "f32", k
            zoom_count += 1
    assert zoom_count == 308
    z1 = json.loads((F / "viewer/scripts/Z1_manip.json").read_text(encoding="utf-8"))
    assert fast.choose_precision(*z1["start"]) == "f32"
    assert fast.choose_precision(10, 0, 50) == "f32"  # extent exactly 60
    assert fast.choose_precision(-10.01, 0, 50) == "f64"
    assert fast.choose_precision(0, -10.01, 50) == "f64"


@pytest.mark.parametrize("scene", ["D1", "Z1", "DZa", "DZb"])
def test_measurement_uses_exact_script_event_states(scene):
    sys.path.insert(0, str(F / "bench"))
    from deepzoom_manip_states import script_states
    for kind in ("manip", "scroll"):
        data = json.loads((F / f"viewer/scripts/{scene}_{kind}.json").read_text(encoding="utf-8"))
        points = list(script_states(data, kind))
        indices = [-1] + (list(range(628)) if kind == "manip" else list(range(59, 600, 60)))
        assert [event for event, _ in points] == indices
        for event, state in points:
            assert vc.same_bits(state, vc.expected_params(data, event))


@pytest.mark.parametrize("low_kb", [0, 8_388_607])
def test_measurement_memory_fallback_and_low_memory_refusal(monkeypatch, low_kb):
    sys.path.insert(0, str(F / "bench"))
    import deepzoom_manip_states as audit
    from types import SimpleNamespace
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(audit, "memory_status_ex", lambda: 8_388_608)
    assert audit.require_host_memory()["query"] == "GlobalMemoryStatusEx"
    def no_fallback():
        raise AssertionError("a successful low CIM query must refuse without fallback")
    monkeypatch.setattr(audit, "memory_status_ex", no_fallback)
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=f"{low_kb}\n"))
    with pytest.raises(RuntimeError, match=f"{low_kb} KB"):
        audit.require_host_memory()
