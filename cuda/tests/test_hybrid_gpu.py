"""CUDA-only hybrid equivalence. Opens no window; run separately from CPU controls."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))


@pytest.fixture(scope="module")
def cuda_only(run_dir):
    # Reuse the existing no-window bench memory guard: CIM query failure falls
    # back to real GlobalMemoryStatusEx; a successfully read low value refuses.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    from deepzoom_manip_states import require_host_memory
    memory = require_host_memory()
    import fast
    info = fast.gpu.device_info()
    (run_dir / "hybrid_cuda_device.json").write_text(json.dumps({"memory": memory, "device": info}, indent=1))
    return fast


@pytest.mark.parametrize("p", [tuple(s["start"]) for s in json.loads(
    (Path(__file__).resolve().parents[1] / "viewer/scripts/deepzoom_scenes.json").read_text(encoding="utf-8"))["scenes"].values()])
def test_fresh_job_is_bit_identical_to_fast_map(cuda_only, p):
    fast = cuda_only
    expected = fast.fast_map(*p)
    job = fast.FreshJob(*p)
    deadline = time.perf_counter() + 30
    actual = job.poll()
    polls = 1
    while actual is None and time.perf_counter() < deadline:
        time.sleep(.001)
        polls += 1
        actual = job.poll()
    assert actual is not None, "FreshJob did not finish within 30s"
    assert job.stream.ptr != 0 and job.precision == fast.choose_precision(*p)
    assert np.array_equal(actual, expected), f"FreshJob mismatch {p}, polls={polls}"
    assert job.poll() is actual


@pytest.mark.parametrize("rows", [32, 64, 96, 128, 192, 256, 485])
def test_chunk_measurement_same_kernel_full_map_and_launch_events(cuda_only, monkeypatch, rows):
    fast = cuda_only
    monkeypatch.setattr(fast, "ROWS_PER_LAUNCH", rows)
    p = (-.465, .918, 4e-9)
    expected = fast.fast_map(*p)
    timing = {}
    job = fast.FreshJob(*p, timing=timing)
    job.event.synchronize()
    actual = job.poll()
    assert actual.shape == (1920,1080) and np.array_equal(actual, expected)
    assert len(timing["launch_ms"]) == (1920 + rows - 1) // rows
    assert all(t >= 0 for t in timing["launch_ms"])


@pytest.mark.parametrize("streams",[1,2,4])
def test_stream_counts_same_32_row_map(cuda_only,monkeypatch,streams):
    fast=cuda_only
    monkeypatch.setattr(fast,"STREAMS",streams)
    monkeypatch.setattr(fast,"ROWS_PER_LAUNCH",32)
    p=(-.465,.918,4e-9)
    expected=fast.fast_map(*p)
    job=fast.FreshJob(*p)
    job.event.synchronize()
    assert np.array_equal(job.poll(),expected)
    low,high=fast.stream_priority_range()
    assert len(job.streams)==streams and all(fast.cp.cuda.runtime.streamGetPriority(s.ptr)==high for s in job.streams)
    assert not job.poll().flags.writeable


def test_extended_axes_screen_bits_and_margin_chunk_kernel(cuda_only):
    fast=cuda_only
    for p in ((0.,0.,50.),(-.465,.918,4e-9),(-.43138,.895038,.1)):
        x,y,_=fast.variants.grid(fast.params.VARIANT,*p,1920)
        xx,yy=fast.extended_axes(*p)
        assert np.array_equal(xx[fast.MX:fast.MX+1920].view(np.uint64),x.view(np.uint64))
        assert np.array_equal(yy[fast.MY:fast.MY+1080].view(np.uint64),y.view(np.uint64))
        expected=fast.fast_map(*p)
        rect=(fast.MX,fast.MX+32,fast.MY,fast.MY+1080)
        chunk=fast.MarginChunk(p,fast.choose_precision(*p),rect)
        chunk.event.synchronize()
        assert np.array_equal(chunk.poll(),expected[:32])
        low,high=fast.stream_priority_range()
        assert fast.cp.cuda.runtime.streamGetPriority(chunk.stream.ptr)==low
