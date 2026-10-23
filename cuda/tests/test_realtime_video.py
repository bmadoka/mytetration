"""viewer/make_realtime_video.py without a window: the scripted tour and the real-time timeline (CPU only)."""
import argparse
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFont

from mytet_check import F

sys.path.insert(0, str(F / "viewer"))
import make_realtime_video as mv  # noqa: E402
import params  # noqa: E402
import video_psnr as vp  # noqa: E402


def test_tour_reaches_z1_with_two_pauses():
    s = mv.tour()
    ev = s["events"]
    t = [e["t"] for e in ev]
    assert t == sorted(t) and 15 <= t[-1] <= 30
    assert all(isinstance(e["dx"], int) and isinstance(e["dy"], int) for e in ev if e["type"] == "drag")
    pauses = [(a, b) for a, b in zip(t, t[1:]) if b - a >= 1.5]  # >= 1.5 s without input: precise frame on screen
    assert len(pauses) == 2 and pauses[-1][1] == t[-1]  # one in the middle, one before the end marker
    assert ev[-1] == {"type": "drag", "dx": 0, "dy": 0, "t": t[-1], "i": len(ev) - 1}
    st = tuple(s["start"])
    for e in ev:
        st = params.apply_event(st, e)
    assert list(st) == s["end_state"]
    x0, y0, eps = st
    assert abs(eps - 0.1) < 0.001 and abs(x0 - mv.TARGET[0]) < 0.01 * eps and abs(y0 - mv.TARGET[1]) < 0.01 * eps


def test_timeline_shows_the_last_presented_frame():
    present = [0.0, 0.010, 0.030, 0.031, 0.100]
    tk, idx = mv.timeline(present, 0.0, 0.1, fps=100)  # output frames at 0, 0.01, ..., 0.09
    assert len(tk) == 10 and idx.tolist() == [0, 1, 1, 2, 3, 3, 3, 3, 3, 3]  # 0.031 is shown from 0.04 on
    vsync = np.array([0, 1, 2, 4, 5, 7]) / 75 + np.array([0, 2e-4, -2e-4, 1e-4, 0, -1e-4])  # presents on a 75 Hz grid
    tk, idx = mv.timeline(vsync, vsync[0], vsync[0] + 8 / 75, fps=75, phase=0.5)
    assert idx.tolist() == [0, 1, 2, 2, 3, 4, 4, 5]  # every presented frame gets its own video frame, held ones repeat


def test_moving_fps_counts_presents_in_the_last_half_second():
    present = np.arange(0, 2, 1 / 60)
    assert abs(mv.moving_fps(present, 1.0) - 60) <= 2
    assert mv.moving_fps(present, 5.0) == 0.0
    assert mv.segments([0.0, 0.1, 0.2, 1.0, 1.1]) == [(0.0, 0.2), (1.0, 1.1)]


@pytest.mark.parametrize("winner", [*range(28, 41), None])
def test_crf_ladder_stops_at_first_fitting_size(winner):
    calls = []

    def size(crf):
        calls.append(crf)
        return 48_000_000 if crf == winner else 48_000_001

    selected, attempts = mv.choose_crf(size, mv.SEND_BYTES, 40)
    expected = list(range(28, 41))
    if winner is not None:
        expected = expected[:expected.index(winner) + 1]
    assert mv.ARCHIVE_BYTES == 62_914_560 and mv.SEND_BYTES == 48_000_000
    assert selected == winner and calls == expected
    assert attempts == [{"crf": crf, "size_bytes": 48_000_000 if crf == winner else 48_000_001} for crf in expected]


def test_psnr_identical_is_infinite():
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    assert vp.luma_psnr(image, image.copy()) == math.inf


def test_psnr_known_noise_uses_float64_luma():
    reference = np.full((2, 2, 3), 128, dtype=np.uint8)
    # Grayscale perturbations have luma errors +10, -10, +20, -20: MSE = 250.
    decoded = (reference.astype(np.int16) + np.array([[10, -10], [20, -20]])[..., None]).astype(np.uint8)
    assert vp.luma_psnr(reference, decoded) == pytest.approx(10 * math.log10(255 ** 2 / 250), abs=1e-12)
    # Channel errors must be weighted before squaring; uint8 subtraction must not wrap.
    reference = np.array([[[255, 0, 0]]], dtype=np.uint8)
    decoded = np.array([[[0, 255, 0]]], dtype=np.uint8)
    assert vp.luma_psnr(reference, decoded) == pytest.approx(20 * math.log10(1 / (0.587 - 0.299)), abs=1e-12)


@pytest.mark.parametrize("bad", [np.zeros((1, 1)), np.zeros((2, 1, 3)), np.zeros((1, 1, 4))])
def test_psnr_rejects_mismatched_rgb_shapes(bad):
    with pytest.raises(ValueError, match="RGB shape"):
        vp.luma_psnr(np.zeros((1, 1, 3)), bad)


@pytest.mark.parametrize("text, expected", [("0", [0]), ("80,0,80, 160 ", [80, 0, 160]), ("001,2", [1, 2])])
def test_frame_index_parsing(text, expected):
    assert vp.parse_frame_indices(text) == expected
    assert mv.parse_frame_indices(text) == expected


@pytest.mark.parametrize("text", ["", " ", "1,", ",1", "1,,2", "-1", "1.0", "x", "+2", "1 2"])
def test_frame_index_parsing_rejects_invalid_input(text):
    with pytest.raises(argparse.ArgumentTypeError):
        vp.parse_frame_indices(text)


@pytest.mark.parametrize("dump_only", [False, True])
def test_reference_png_matches_encoder_bytes_and_dump_only_skips_encoding(monkeypatch, dump_only):
    # All I/O is in memory; the existing composition runs on synthetic captured bits.
    meta = {"recorded_frames": 2, "recorded_non_binary_pixels": 0, "input_t": {}, "t_end": 1 + 3 / 75,
            "monitor_refresh_hz": 75}
    frames = [{"t_present": 1 + i / 75, "kind": "fast", "present_kind": "warped", "precision": "f64"}
              for i in range(2)]
    bits = np.packbits(np.arange(2 * mv.W * mv.H, dtype=np.uint8).reshape(2, mv.H, mv.W) % 3 == 0, axis=None)
    bits = bits.reshape(2, -1)
    font = ImageFont.load_default()
    commands, piped, saved, summaries, renamed, checked = [], [], {}, {}, [], []
    original_save, original_open = Image.Image.save, Image.open

    def save_png(im, path, **kwargs):
        buffer = io.BytesIO()
        original_save(im, buffer, format="PNG")
        with original_open(io.BytesIO(buffer.getvalue())) as png:
            saved[Path(path).name] = np.asarray(png).copy()

    class Pipe(io.BytesIO):
        def close(self):
            piped.append(self.getvalue())
            super().close()

    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdin=Pipe(), wait=lambda: 0)

    with monkeypatch.context() as m:
        m.setattr(mv, "load", lambda path: (meta, frames))
        m.setattr(mv.np, "load", lambda path: bits)
        m.setattr(Path, "read_text", lambda *args, **kwargs: json.dumps({"events": [{"type": "drag", "dx": 0, "dy": 0}]}))
        m.setattr(ImageFont, "truetype", lambda *args, **kwargs: font)
        m.setattr(Image.Image, "save", save_png)
        m.setattr(Path, "mkdir", lambda *args, **kwargs: None)
        m.setattr(Path, "exists", lambda path: dump_only)  # existing MP4 allowed for dump-only
        m.setattr(Path, "stat", lambda path, **kwargs: SimpleNamespace(st_size=2**20))
        m.setattr(mv.shutil, "copyfile", lambda src, dst: renamed.append((src.name, dst.name)))
        m.setattr(Path, "write_text", lambda path, text, **kwargs: summaries.update({path.name: json.loads(text)}))
        m.setattr(mv.subprocess, "Popen", popen)
        m.setattr(mv, "checks", lambda *args: checked.append(True) or {})
        result = mv.video(F / "synthetic_run", [], dump_ref=[0, 1, 2], dump_only=dump_only)
    assert set(saved) == {"frame_0.png", "frame_1.png", "frame_2.png"}
    if dump_only:
        assert result == {"dump_only": True, "video_frames": 3, "dumped_ref_frames": [0, 1, 2]}
        assert not commands and not piped and not summaries and not renamed and not checked
    else:
        assert len(commands) == 1 and checked == [True, True]
        command = commands[0]
        for flag, value in (("-c:v", "libx264"), ("-preset", "slow"), ("-crf", "28"), ("-r", "75.0"),
                            ("-movflags", "+faststart")):
            assert command[command.index(flag) + 1] == value
        assert command[command.index("-pix_fmt") + 1] == "rgb24"
        assert command[command.index("-crf") + 3] == "yuv420p"
        assert "-an" in command and "-pass" not in command and "-b:v" not in command
        assert piped == [b"".join(saved[f"frame_{n}.png"].tobytes() for n in range(3))]
        assert renamed == [("_encode_crf28.mp4", "viewer_realtime_archive.mp4"),
                           ("_encode_crf28.mp4", "viewer_realtime_send.mp4")]
        outputs = summaries["summary.json"]["outputs"]
        assert outputs["archive"]["crf"] == outputs["send"]["crf"] == 28
        for item in outputs.values():
            assert item["size_bytes"] == 2**20
            assert item["encode_attempts"] == [{"crf": 28, "size_bytes": 2**20, "file": "_encode_crf28.mp4"}]


def test_encoding_refuses_an_existing_mp4_before_loading(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(mv, "load", lambda path: pytest.fail("must refuse before loading"))
    with pytest.raises(RuntimeError, match="exists"):
        mv.video(F / "synthetic_run", [])


def test_decode_selects_zero_based_frame_and_rgb24(monkeypatch):
    expected = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"capture_output": True, "check": True}
        return SimpleNamespace(stdout=expected.tobytes())

    monkeypatch.setattr(vp.subprocess, "run", run)
    assert np.array_equal(vp.decode_frame("test.mp4", 80, expected.shape), expected)
    command = commands[0]
    assert command[0] == vp.FFMPEG
    assert command[command.index("-vf") + 1] == "select=eq(n\\,80)"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"
    monkeypatch.setattr(vp.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=b""))
    with pytest.raises(ValueError, match="missing"):
        vp.decode_frame("test.mp4", 999, expected.shape)


def test_psnr_cli_table_json_and_comparison_are_consistent(monkeypatch, capsys):
    reference = np.full((2, 3, 3), 100, dtype=np.uint8)
    saved, written = {}, {}
    monkeypatch.setattr(Image, "open", lambda path: Image.fromarray(reference))
    monkeypatch.setattr(vp, "decode_frame", lambda mp4, index, shape: reference + (0 if mp4.name == "a.mp4" else 10))
    monkeypatch.setattr(Image.Image, "save", lambda im, path: saved.update({str(path): np.asarray(im).copy()}))
    monkeypatch.setattr(Path, "write_text", lambda path, text, **kwargs: written.update({path.name: text}))
    result = vp.main(["a.mp4", "b.mp4", "--ref-dir", "refs", "--frames", "0,80", "--compare", "80",
                      "--compare-out", "comparison.png"])
    assert json.loads(written["video_psnr.json"]) == result
    assert result["videos"][0]["psnr_db"] == {"0": "inf", "80": "inf"}
    assert result["videos"][1]["psnr_db"]["80"] == pytest.approx(20 * math.log10(255 / 10))
    assert np.array_equal(saved["comparison.png"], np.concatenate([reference, reference, reference + 10], axis=1))
    table = capsys.readouterr().out
    assert "| frame | a.mp4 | b.mp4 |" in table and "| 80 | inf | 28.1308 |" in table


def test_binary_mismatch_excludes_inclusive_box_and_uses_luma_threshold():
    ref = np.zeros((10, 10, 3), np.uint8)
    decoded = ref.copy()
    decoded[1:4, 2:6] = 255
    assert vp.binary_mismatch(ref, decoded, (2,1,5,3)) == 0
    decoded[0, 0] = 128
    assert vp.binary_mismatch(ref, decoded, (2,1,5,3)) == pytest.approx(1/88)
    decoded[0, 0] = 127
    assert vp.binary_mismatch(ref, decoded, (2,1,5,3)) == 0
    decoded[0, 0] = [255, 0, 0]  # red luma is below 127
    assert vp.binary_mismatch(ref, decoded, (2,1,5,3)) == 0
    summary = vp.mismatch_summary({"0": .02, "1": .021, "2": .03, "3": .031})
    assert summary["frames_over_2_percent"] == [1,2,3]
    assert summary["frames_over_3_percent"] == [3]
    assert summary["max"] == .031 and summary["median"] == pytest.approx(.0255)


def test_all_frames_recomposes_and_decodes_once_per_video(monkeypatch, tmp_path):
    refs = [np.full((2,3,3), v, np.uint8) for v in (0,255,0)]
    decoded_calls, composed_calls = [], []
    def compose(run):
        composed_calls.append(run)
        yield from refs
    def decode(mp4):
        decoded_calls.append(mp4)
        yield from [refs[0], np.zeros_like(refs[1]), refs[2]]
    monkeypatch.setattr(mv, "iter_composed_frames", compose)
    monkeypatch.setattr(vp, "decode_frames", decode)
    result = vp.main(["a.mp4", "--all-frames", "--recording", "recording", "--out", str(tmp_path / "psnr.json")])
    assert len(decoded_calls) == len(composed_calls) == 1
    assert result["frames"] == [0,1,2] and result["overlay_box_inclusive"] == list(mv.OVERLAY_BOX)
    score = result["videos"][0]
    assert score["binary_summary"]["frames_over_3_percent"] == [1]
    assert score["binary_summary"]["max"] == 1.0 and score["binary_summary"]["median"] == 0
    monkeypatch.setattr(vp, "decode_frames", lambda path: iter(refs[:2]))
    with pytest.raises(ValueError, match="counts differ"):
        vp.main(["a.mp4", "--all-frames", "--recording", "recording", "--out", str(tmp_path / "bad.json")])


def test_single_pass_decoder_yields_frames_and_rejects_partial(monkeypatch):
    frames = np.arange(36, dtype=np.uint8).reshape(2,2,3,3)
    commands = []
    def process(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=io.BytesIO(frames.tobytes()), stderr=io.BytesIO(),
                               wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr(vp.subprocess, "Popen", process)
    assert np.array_equal(np.stack(list(vp.decode_frames("a.mp4", (2,3,3)))), frames)
    assert len(commands) == 1 and "-vf" not in commands[0]


def test_archive_and_send_search_use_separate_exact_caps():
    sizes = {28: 62_914_561, 29: 62_914_560, 30: 48_000_001, 31: 48_000_000}
    assert mv.choose_crf(sizes.__getitem__)[0] == 29
    assert mv.choose_crf(sizes.__getitem__, mv.SEND_BYTES, 40)[0] == 31
    selected, attempts = mv.choose_crf(lambda c: 48_000_001, mv.SEND_BYTES, 40)
    assert selected is None and [r["crf"] for r in attempts] == list(range(28,41))


def test_exposed_composition_keeps_original_overlay_labels(monkeypatch):
    from PIL import ImageDraw
    texts = []
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", lambda self, xy, text, **kw: texts.append(text))
    font = ImageFont.load_default()
    packed = np.packbits(np.zeros((mv.H, mv.W), bool))
    rgb = mv.compose_frame(packed, {"kind":"fast", "present_kind":"warped", "precision":"f64"},
                           1., [0.,1.], [1.], 0., (font,font))
    assert texts == ["실시간 1×    1.00 s", "fps   2.0  (최근 0.5 s)", "옮김 · f64", "입력 중"]
    assert rgb.shape == (540,960,3) and rgb.dtype == np.uint8
