"""Recording compatibility and job ownership controls; CPU only."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import viewer_check as vc
from test_hybrid_cpu import b2_cpu_viewer, duplicate_launch_records
import hybrid

F = Path(__file__).resolve().parents[1]


def local_only_record(name):
    """Per-run measurement records are not part of this package: skip visibly where they are absent."""
    d = F / "logs/hybrid/measure_timing" / name
    if not (d / "meta.json").is_file():
        pytest.skip(f"measurement record not included: {d.relative_to(F).as_posix()}")
    return d


@pytest.mark.parametrize("index", [0, 1])
def test_flipping_async_cannot_hide_duplicate_or_relabel_sync(index):
    frames, jobs = duplicate_launch_records()
    assert vc.duplicate_job_check(frames, jobs)["count"] == 1
    jobs[index]["async"] = not jobs[index]["async"]
    with pytest.raises(ValueError, match="job record error.*synchronous ownership"):
        vc.duplicate_job_check(frames, jobs)


def test_zero_duration_sync_is_identified_by_its_recorded_map():
    frames, jobs = duplicate_launch_records()
    jobs[0]["t_end"] = frames[1]["t_compute_end"] = 0.
    jobs[0]["sha256"] = frames[1]["map_sha256"] = "same"
    assert vc.duplicate_job_check(frames, jobs)["count"] == 1
    jobs[0]["async"] = True
    with pytest.raises(ValueError, match="job record error.*synchronous ownership"):
        vc.duplicate_job_check(frames, jobs)


@pytest.mark.parametrize("field,value", [
    ("params", (-0., 0., 1.)), ("t_start", -.001), ("t_end", .039), ("sha256", "different")])
def test_sync_ownership_requires_exact_identity(field, value):
    frames, jobs = duplicate_launch_records()
    frames[1]["map_sha256"] = jobs[0]["sha256"] = "same"
    jobs[0][field] = value
    with pytest.raises(ValueError, match="job record error"):
        vc.duplicate_job_check(frames, jobs)


@pytest.mark.parametrize("mutation", ["missing_job", "two_jobs", "two_presents"])
def test_each_sync_present_and_job_has_exactly_one_owner(mutation):
    frames, jobs = duplicate_launch_records()
    if mutation == "missing_job":
        jobs.pop(0)
    elif mutation == "two_jobs":
        jobs.append(dict(jobs[0], id=2))
    else:
        frames.append(dict(frames[1], seq=2))
    with pytest.raises(ValueError, match="job record error"):
        vc.duplicate_job_check(frames, jobs)


def test_real_sync_switching_frame_and_async_tail(monkeypatch):
    v, clock = b2_cpu_viewer(monkeypatch, .04)
    v.frame("fast", clock[0])
    assert v.frames[0]["mode"] == "warped" and not v.fresh[0]["async"]
    assert vc.duplicate_job_check(v.frames, v.fresh)["count"] == 0
    clock[0] += .02
    v.state = (1., 0., 50.)
    v.applied += 1
    v.frame("fast", clock[0])
    clock[0] += .05
    v._complete_fresh(v.source_map, v.job_record)
    assert v.fresh[-1]["t_end"] > v.frames[-1]["t_present"]
    assert vc.duplicate_job_check(v.frames, v.fresh)["count"] == 0


@pytest.mark.parametrize("name,duplicates", [("D1_manip_run2", 0), ("DZb_scroll_run2", 1)])
def test_real_historical_recordings_check_without_gpu(name, duplicates):
    d = local_only_record(name)
    meta, frames, _ = vc.load(d)
    fresh, _ = vc.load_fresh(d)
    assert meta["hybrid_version"] == 1 and all("mode" not in r for r in fresh.values())
    result = vc.check_run(d, recheck=False, precise_check=False)
    assert result["hybrid"]["duplicate_jobs"]["count"] == duplicates
    assert not any(result["hybrid"]["records"].values())
    assert not result["param_mismatch"]
    assert not result["hybrid"]["condition4"]["bad"]


def test_legacy_duplicate_recount_derives_mode_from_present():
    frames, jobs = duplicate_launch_records()
    for job in jobs:
        del job["mode"]
    assert vc.duplicate_job_check(frames, jobs, version=1)["count"] == 1


def test_legacy_reconstruction_matches_historical_single_source():
    # Execute only the historical pure CPU function, never its launch/GPU entry points.
    path = F / "bench/hybrid_b2/legacy/viewer_check_v1.py.txt"  # historical checker source, not included
    if not path.is_file():
        pytest.skip(f"historical source not included: {path.relative_to(F).as_posix()}")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("displayed_map", "independent_warp")]
    scope = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "viewer_check_v1", "exec"), scope)
    p = (0., 0., 50.)
    maps = {0: np.random.default_rng(3).integers(0, 2, (1920, 1080)).astype(bool)}
    fresh = {0: {"id": 0, "params": p}}
    frame = {"source_fresh_id": 0, "display_params": (1., 0., 50.), "present_kind": "warped",
             "candidate_ids": [999]}  # Version 1 never selects B2, even if unrelated fields exist.
    assert np.array_equal(vc.displayed_map(frame, fresh, maps, {"hybrid_version": 1}),
                          scope["displayed_map"](frame, fresh, maps))


@pytest.fixture
def b2_recording(tmp_path):
    d = local_only_record("D1_manip_run2")
    meta, frames, _ = vc.load(d)
    fresh, maps = vc.load_fresh(d)
    frames = copy.deepcopy(frames[:2])
    job = dict(fresh[0], mode="fresh-only")
    meta.update(hybrid_version=2, fast_calls_playback=1, precision_calls_playback=[job["precision"]])
    f = frames[1]
    f["candidate_ids"] = f["bound_candidate_ids"] = [0]
    _, f["candidate_transforms"], f["pixel_counts"] = hybrid.candidate_plan(
        f["display_params"], [job])
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "frames.jsonl").write_text("\n".join(map(json.dumps, frames)), encoding="utf-8")
    (tmp_path / "fresh.jsonl").write_text(json.dumps(job), encoding="utf-8")
    np.save(tmp_path / "fresh_maps.npy", np.packbits(maps[0])[None, :])
    result = vc.check_run(tmp_path, recheck=False, precise_check=False)
    assert result["hybrid"]["duplicate_jobs"]["count"] == 0
    assert not any(result["hybrid"]["records"].values())
    return tmp_path


def test_version2_all_candidate_fields_stripped_fails(b2_recording, monkeypatch):
    d = b2_recording
    frames = [json.loads(line) for line in (d / "frames.jsonl").read_text().splitlines()]
    for frame in frames:
        for key in vc.CANDIDATE_FIELDS:
            frame.pop(key, None)
    (d / "frames.jsonl").write_text("\n".join(map(json.dumps, frames)), encoding="utf-8")
    with pytest.raises(ValueError, match="B2 present record error"):
        vc.check_run(d, recheck=False, precise_check=False)
    import launch
    monkeypatch.setattr(launch, "run", lambda *a, **k: pytest.fail("invalid record reached window launcher"))
    with pytest.raises(ValueError, match="B2 present record error"):
        vc.check_replay(d)


def test_version2_job_missing_mode_fails(b2_recording, monkeypatch):
    p = b2_recording / "fresh.jsonl"
    job = json.loads(p.read_text())
    del job["mode"]
    p.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(ValueError, match="job record error"):
        vc.check_run(b2_recording, recheck=False, precise_check=False)
    import launch
    monkeypatch.setattr(launch, "run", lambda *a, **k: pytest.fail("invalid job reached window launcher"))
    with pytest.raises(ValueError, match="job record error"):
        vc.check_replay(b2_recording)


@pytest.mark.parametrize("version", [None, 0, 4, "2", True])
def test_missing_or_unknown_recording_version_is_error(b2_recording, version):
    p = b2_recording / "meta.json"
    meta = json.loads(p.read_text())
    if version is None:
        del meta["hybrid_version"]
    else:
        meta["hybrid_version"] = version
    p.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="record error.*hybrid_version"):
        vc.check_run(b2_recording, recheck=False, precise_check=False)


def test_writer_marker_is_version3_without_constructing_viewer():
    tree = ast.parse((F / "viewer/app.py").read_text(encoding="utf-8"))
    markers = [value.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
               for key, value in zip(node.keys, node.values)
               if isinstance(key, ast.Constant) and key.value == "hybrid_version"]
    assert markers == [3]
