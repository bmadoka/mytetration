"""Real-time video of the viewer: two MP4s of one scripted session at its real speed.

Frames are the viewer's own: in a timed replay with --readback record the viewer reads its window back buffer right
before every present (960 x 540, black / white) and saves them with the frame log; no screen capture, no OS input.
The video time axis is the frames' real present times: a video at the display refresh rate fps (the recording's
monitor_refresh_hz) whose frame k shows the last frame presented at or before (k + 0.5)/fps s after the first present
(no speed-up, no interpolation; a frame
that stayed on screen for several refreshes is repeated, as on the monitor). Overlay (top left, small): «실시간 1×» + time,
fps = frames presented in the last 0.5 s / 0.5 s of that recorded run, the mode of the frame on screen
(새 · f32/f64 = a freshly computed fast map, 옮김 · f32/f64 = a warped earlier map, 정밀 · double) and whether input
is arriving.

usage: python viewer/make_realtime_video.py script                -> preview_media/hybrid/realtime_tour.json
       python viewer/make_realtime_video.py run [--out DIR]       -> runs none, record, none, record
       python viewer/make_realtime_video.py video RUN_DIR         -> viewer_realtime_archive.mp4 / viewer_realtime_send.mp4 + checks
"""
import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(F))
import params  # noqa: E402
sys.path.insert(0, str(F / "scripts"))
from video_psnr import parse_frame_indices  # noqa: E402

OUT = F / "preview_media" / "hybrid"
SCRIPT = OUT / "realtime_tour.json"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
FONT = os.path.join(os.environ.get("WINDIR", ""), "Fonts", "malgun.ttf")  # Windows Korean font (Malgun Gothic)
FPS = 75                          # fallback only, for a recording without monitor_refresh_hz (presents are paced by DwmFlush)
W, H = 960, 540
ARCHIVE_BYTES = 62_914_560
SEND_BYTES = 48_000_000
OVERLAY_BOX = (10, 10, 222, 112)  # inclusive PIL rectangle; binary comparison excludes it
START = (0.0, 0.0, 50.0)          # the 16:9 default view (scene D1)
TARGET = (-0.43138, 0.895038)     # centre of scene Z1
RATE = 60                         # script events per second while moving


def _drag_to(state, cx, cy):
    """Integer render-target drag that moves the view centre of `state` closest to (cx, cy)."""
    x0, y0, eps = state
    return round((x0 - cx) * 1919 / (2 * eps)), round((cy - y0) * 1079 / (2 * params.eps_y(eps)))


def tour():
    """The scripted session (times in s after playback start; the initial precise frame is on screen before 0)."""
    events, st = [], START

    def add(t, e):
        nonlocal st
        events.append(dict(e, t=round(t, 6)))
        st = params.apply_event(st, e)

    def loop(t0, seconds, radius):  # drag around a circle of `radius` render-target px and back to the start
        n, prev = int(seconds * RATE), (0, 0)
        for i in range(1, n + 1):
            th = 2 * math.pi * i / n
            X, Y = (0, 0) if i == n else (round(radius * (math.cos(th) - 1)), round(radius * math.sin(th)))
            if (X - prev[0], Y - prev[1]) != (0, 0):
                add(t0 + i / RATE, {"type": "drag", "dx": X - prev[0], "dy": Y - prev[1]})
            prev = (X, Y)

    def zoom(t0, seconds, notches, centre_of):  # wheel in, and drag so the centre follows centre_of(eps)
        n = int(seconds * RATE)
        for i in range(1, n + 1):
            add(t0 + i / RATE, {"type": "wheel", "steps": notches / n})
            dx, dy = _drag_to(st, *centre_of(st[2]))
            if (dx, dy) != (0, 0):
                add(t0 + i / RATE, {"type": "drag", "dx": dx, "dy": dy})

    loop(1.0, 3.0, 300)                                                   # A 1.0-4.0 s: drag the picture around
    c0, e0 = st[:2], st[2]
    zoom(4.0, 3.5, 37, lambda e: (TARGET[0] + (c0[0] - TARGET[0]) * e / e0,  # B 4.0-7.5 s: zoom in towards Z1
                                  TARGET[1] + (c0[1] - TARGET[1]) * e / e0))  # (Z1 stays where it is on screen)
    c1, e1 = st[:2], st[2]                                                # pause 7.5-9.5 s: precise frame
    zoom(9.5, 2.5, 22, lambda e: (TARGET[0] + (c1[0] - TARGET[0]) * (e / e1) ** 2,  # C 9.5-12.0 s: further in,
                                  TARGET[1] + (c1[1] - TARGET[1]) * (e / e1) ** 2))  # centre onto Z1
    loop(12.0, 3.0, 250)                                                  # D 12.0-15.0 s: move around and back
    add(18.0, {"type": "drag", "dx": 0, "dy": 0})  # end marker: no change, no frame; keeps the window 3 s on precise
    for k, e in enumerate(events):
        e["i"] = k
    return {"scene": "realtime_tour", "start": list(START), "events": events, "end_state": list(st)}


# ---------------- runs ----------------
def gpu_line():
    from mytet_cuda import device
    return subprocess.run(["nvidia-smi", f"--id={device.choose()['uuid']}", "--query-gpu=timestamp,name,"
                           "memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def runs(out):
    import launch
    out.mkdir(parents=True, exist_ok=False)
    log = open(out / "runs_log.txt", "a", encoding="utf-8")

    def say(msg):
        msg = f"{dt.datetime.now():%H:%M:%S} {msg}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    for k, rb in enumerate(("none", "record", "none", "record")):
        name = f"{rb}_{k // 2 + 1}"
        say(f"{name}: GPU {gpu_line()}")
        from cpu_guard import guarded_run
        d = guarded_run(lambda directory: launch.run(SCRIPT,"timed",directory,readback=rb),out/name)
        contamination=json.loads((d/"contamination.json").read_text(encoding="utf-8"))
        say(f"{name}: contamination {contamination['verdict']}")
        say(f"{name}: done {json.loads((d / 'launch.json').read_text(encoding='utf-8'))['wall_s']:.1f} s")
    log.close()


# ---------------- analysis ----------------
def load(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    frames = [json.loads(s) for s in (d / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    return meta, frames


def segments(times, gap=0.25):
    """Groups of input times with gaps < `gap` -> [(first, last)]."""
    seg = []
    for t in sorted(times):
        if seg and t - seg[-1][1] < gap:
            seg[-1][1] = t
        else:
            seg.append([t, t])
    return [tuple(s) for s in seg]


def moving_fps(present, t, window=0.5):
    """Frames presented in (t - window, t] divided by window."""
    p = np.asarray(present)
    return float(np.count_nonzero((p > t - window) & (p <= t)) / window)


def fps_stats(meta, frames, end_marker):
    """Per motion segment: fast frames / (last fast present - first input); min of the 0.5 s moving average at
    every fast present at least 0.5 s into the segment."""
    inputs = {int(k): v for k, v in meta["input_t"].items() if int(k) != end_marker}
    present = [f["t_present"] for f in frames]
    out = []
    for a, b in segments(inputs.values()):
        fast = [f["t_present"] for f in frames if f["kind"] == "fast" and a <= f["t_present"] <= b + 0.25]
        if len(fast) < 2:
            continue
        mv = [moving_fps(present, t) for t in fast if t - a >= 0.5]
        out.append({"first_input": a - meta["t_playback_start"], "last_input": b - meta["t_playback_start"],
                    "fast_frames": len(fast), "fps": len(fast) / (fast[-1] - a),
                    "fps_moving_min": min(mv) if mv else None,
                    "frame_time_p95_ms": float(np.percentile(np.diff(fast), 95) * 1000)})
    if not out and inputs:  # slow frames: input batches look like separate segments -> one segment over all inputs
        a, b = min(inputs.values()), max(inputs.values())
        fast = [f["t_present"] for f in frames if f["kind"] == "fast" and a <= f["t_present"]]
        if len(fast) >= 2:
            mv = [moving_fps(present, t) for t in fast if t - a >= 0.5]
            out.append({"first_input": a - meta["t_playback_start"], "last_input": b - meta["t_playback_start"],
                        "fast_frames": len(fast), "fps": len(fast) / (fast[-1] - a),
                        "fps_moving_min": min(mv) if mv else None,
                        "frame_time_p95_ms": float(np.percentile(np.diff(fast), 95) * 1000), "whole_run": True})
    total = sum(s["fast_frames"] for s in out)
    span = sum(s["fast_frames"] / s["fps"] for s in out)
    mins = [s["fps_moving_min"] for s in out if s["fps_moving_min"] is not None]
    return {"segments": out, "fps_mean": total / span if span else None, "fps_moving_min": min(mins) if mins else None}


def timeline(present, t_begin, t_end, fps=FPS, phase=0.0):
    """Output frame k (video time k/fps) -> index of the last frame presented at or before t_begin + (k + phase)/fps.
    With fps = the display refresh and phase 0.5 every sample sits between two refreshes, so each presented
    frame (one per refresh at most, presents are DwmFlush-paced) lands in its own video frame."""
    n = int(round((t_end - t_begin) * fps))
    tk = t_begin + (np.arange(n) + phase) / fps
    return tk, np.searchsorted(np.asarray(present), tk, side="right") - 1


def frame_label(frame):
    if frame["kind"] != "fast":
        return "\uc815\ubc00 \u00b7 double"
    label = "\uc62e\uae40" if frame.get("present_kind") == "warped" else "\uc0c8"
    return f"{label} \u00b7 {frame.get('precision', 'f32')}"


def recording_timeline(meta, frames):
    """Preserve present times and give the final present at least one output frame.

    If the window closed immediately after its final swap, the appended hold is
    recorded explicitly; it is not counted as measured on-screen time.
    """
    present = [f["t_present"] for f in frames]
    fps = float(meta.get("monitor_refresh_hz", FPS))
    begin = present[0]
    end = max(meta["t_end"], present[-1] + 1.5 / fps)
    times, indices = timeline(present, begin, end, fps=fps, phase=.5)
    return times, indices, fps, begin, end, end - meta["t_end"]


def choose_crf(size_bytes_for_crf, max_bytes=ARCHIVE_BYTES, max_crf=51):
    """Lowest CRF starting at 28 fitting an exact byte cap; preserve all attempts."""
    attempts = []
    for crf in range(28, max_crf + 1):
        size = int(size_bytes_for_crf(crf))
        attempts.append({"crf": crf, "size_bytes": size})
        if size <= max_bytes:
            return crf, attempts
    return None, attempts


def recording_inputs(meta):
    script = json.loads(Path(meta.get("script", SCRIPT)).read_text(encoding="utf-8"))
    last = script["events"][-1]
    marker = last.get("type") == "drag" and last.get("dx") == 0 and last.get("dy") == 0
    end_marker = len(script["events"]) - 1 if marker else -1
    return sorted(v for k, v in meta["input_t"].items() if int(k) != end_marker), end_marker


def compose_frame(packed_bits, frame, t, present, inputs, t_begin, fonts=None):
    """Exact RGB bytes shared by the encoder and independent video comparisons."""
    from PIL import Image, ImageDraw, ImageFont
    if fonts is None:
        fonts = (ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT.replace("malgun", "malgunbd"), 18))
    font, font_b = fonts
    img = np.unpackbits(packed_bits)[:W * H].reshape(H, W)[::-1] * np.uint8(255)
    im = Image.fromarray(np.repeat(img[..., None], 3, axis=2)).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    dr.rounded_rectangle(OVERLAY_BOX, radius=6, fill=(0, 0, 0, 215))
    k = np.searchsorted(inputs, t, side="right")
    since = t - inputs[k - 1] if k else None
    precise = frame["kind"] != "fast"
    dr.text((20, 14), f"실시간 1×   {t - t_begin:5.2f} s", font=font, fill=(255, 255, 255, 255))
    dr.text((20, 37), f"fps {moving_fps(present, t):5.1f}  (최근 0.5 s)", font=font, fill=(255, 255, 255, 255))
    dr.text((20, 60), frame_label(frame), font=font_b,
            fill=(90, 220, 255, 255) if precise else (255, 170, 60, 255))
    dr.text((20, 85), "입력 전" if since is None else ("입력 중" if since < 0.1 else f"손 뗌 {since:4.2f} s"),
            font=font, fill=(200, 200, 200, 255))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def iter_composed_frames(run_dir):
    """Stream all references from the same composition/timeline as video(); no window/GPU."""
    from PIL import ImageFont
    meta, frames = load(run_dir)
    bits = np.load(Path(run_dir) / "window_frames_bits.npy", mmap_mode="r")
    if len(bits) != len(frames) or meta["recorded_non_binary_pixels"] != 0:
        raise ValueError("invalid binary recording")
    inputs, _ = recording_inputs(meta)
    times, indices, _, begin, _, _ = recording_timeline(meta, frames)
    fonts = (ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT.replace("malgun", "malgunbd"), 18))
    present = [f["t_present"] for f in frames]
    for t, i in zip(times, indices):
        yield compose_frame(bits[i], frames[i], t, present, inputs, begin, fonts)


def video(run_dir, none_dirs, dump_ref=(), dump_only=False):
    from PIL import Image, ImageFont
    destinations = {name: OUT / f"viewer_realtime_{name}.mp4" for name in ("archive", "send")}
    if not dump_only and any(p.exists() for p in destinations.values()):
        raise RuntimeError("archive/send MP4 exists; choose a new output directory")
    if dump_only and not dump_ref:
        raise ValueError("--dump-only requires --dump-ref")
    meta, frames = load(run_dir)
    inputs, end_marker = recording_inputs(meta)
    bits = np.load(Path(run_dir) / "window_frames_bits.npy")
    assert len(bits) == len(frames) == meta["recorded_frames"] and meta["recorded_non_binary_pixels"] == 0
    present = [f["t_present"] for f in frames]
    tk, idx, video_fps, t_begin, t_end, final_hold = recording_timeline(meta, frames)
    if any(n < 0 or n >= len(idx) for n in dump_ref):
        raise ValueError(f"reference indices must be in 0..{len(idx) - 1}")
    font = ImageFont.truetype(FONT, 17)
    font_b = ImageFont.truetype(FONT.replace("malgun", "malgunbd"), 18)
    fonts = (font, font_b)
    out = [compose_frame(bits[i], frames[i], t, present, inputs, t_begin, fonts) for t, i in zip(tk, idx)]
    OUT.mkdir(parents=True, exist_ok=True)
    if dump_ref:
        ref_dir = OUT / "ref_frames"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for n in dump_ref:
            Image.fromarray(out[n]).save(ref_dir / f"frame_{n}.png")
    if dump_only:
        return {"dump_only": True, "video_frames": len(out), "dumped_ref_frames": list(dump_ref)}

    def encode(crf):
        dst = OUT / f"_encode_crf{crf}.mp4"
        p = subprocess.Popen([FFMPEG, "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(video_fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
                              "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                             stdin=subprocess.PIPE)
        try:
            for fr in out:
                p.stdin.write(fr.tobytes())
        finally:
            p.stdin.close()
            returncode = p.wait()
        if returncode != 0:
            raise RuntimeError("ffmpeg failed")
        return dst.stat().st_size

    encoded = {}

    def cached_encode(crf):
        if crf not in encoded:
            encoded[crf] = encode(crf)
        return encoded[crf]

    outputs = {}
    for name, cap, limit in (("archive", ARCHIVE_BYTES, 51), ("send", SEND_BYTES, 40)):
        selected, attempts = choose_crf(cached_encode, cap, limit)
        for attempt in attempts:
            attempt["file"] = f"_encode_crf{attempt['crf']}.mp4"
        outputs[name] = {"file": destinations[name].name, "cap_bytes": cap, "crf": selected,
                         "encode_attempts": attempts}
        if selected is None:
            failure = {"outputs": outputs, "error": f"no CRF in 28..{limit} fits {name} cap {cap}"}
            (OUT / "summary.json").write_text(json.dumps(failure, indent=1), encoding="utf-8")
            raise RuntimeError(failure["error"])
        shutil.copyfile(OUT / f"_encode_crf{selected}.mp4", destinations[name])
        outputs[name]["size_bytes"] = encoded[selected]
    # precise transitions (the moment a precise frame is presented) in video time
    trans = []
    for f in frames:
        if f["kind"] != "fast":
            last = max([v for v in inputs if v <= f["t_present"]], default=None)
            trans.append({"video_s": f["t_present"] - t_begin, "kind": f["kind"],
                          "after_last_input_s": None if last is None else f["t_present"] - last})
    res = {"run": str(Path(run_dir).resolve().relative_to(F)), "outputs": outputs, "overlay_box_inclusive": list(OVERLAY_BOX),
           "frames_presented": len(frames),
           "frames_shown_in_video": int(len(set(idx.tolist()))), "video_frames": len(out),
           "real_elapsed_s": meta["t_end"] - t_begin, "display_timeline_s": t_end - t_begin,
           "final_frame_hold_s": final_hold, "video_s_expected": len(out) / video_fps, "precise_transitions": trans,
           "fps_record": fps_stats(meta, frames, end_marker),
           "fps_none": {str(Path(d).resolve().relative_to(F)): fps_stats(*load(d), end_marker) for d in none_dirs}}
    final_precise = next((f for f in reversed(frames) if f["kind"] == "precise"), None)
    if final_precise is not None:
        sys.path.insert(0, str(F / "tests"))
        import viewer_check as vc
        vc.require_memory()
        import fast
        pm = fast.precise_map(*final_precise["params"])
        Image.fromarray(vc.picture(pm)[::-1, :, :3]).save(OUT / "precise.png")
        res["precise_png"] = "precise.png"
    for name, destination in destinations.items():
        outputs[name]["checks"] = checks(destination, res)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def checks(mp4, res):
    pr = json.loads(subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-show_entries",
                                    "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:"
                                    "format=duration,size", "-of", "json", str(mp4)],
                                   capture_output=True, text=True, check=True).stdout)
    dec = subprocess.run([FFMPEG, "-v", "error", "-i", str(mp4), "-f", "null", "-"], capture_output=True, text=True)
    v = [s for s in pr["streams"] if s["codec_type"] == "video"]
    dur = float(pr["format"]["duration"])
    n = int(v[0]["nb_read_frames"])
    pngs = {}
    for name, k in (("first", 0), ("middle", n // 2), ("last", n - 1)):
        png = OUT / f"{mp4.stem}_frame_{name}.png"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(mp4), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1",
                        str(png)], check=True)
        pngs[name] = {"frame": k, "file": png.name}
    return {"ffprobe": pr, "audio_streams": sum(s["codec_type"] == "audio" for s in pr["streams"]),
            "decode_errors": dec.stderr.strip().splitlines(), "decode_returncode": dec.returncode,
            "duration_s": dur, "frames": n, "size_mb": int(pr["format"]["size"]) / 2**20,
            "video_vs_real_diff_pct": 100 * abs(dur - res["real_elapsed_s"]) / res["real_elapsed_s"], "png": pngs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["script", "run", "video"])
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--none", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--dump-ref", type=parse_frame_indices, default=[], help="zero-based video indices, e.g. 0,80,160")
    ap.add_argument("--dump-only", action="store_true", help="write references without encoding or checks")
    ap.add_argument("--script", help="script of the recorded run (default: the realtime tour)")
    ap.add_argument("--media-out", help="folder for the MP4 / PNGs / summary (default: preview_media/hybrid)")
    a = ap.parse_args(argv)
    if (a.dump_ref or a.dump_only) and a.cmd != "video":
        ap.error("--dump-ref / --dump-only require the video command")
    if a.dump_only and not a.dump_ref:
        ap.error("--dump-only requires --dump-ref")
    global SCRIPT, OUT
    if a.script:
        SCRIPT = Path(a.script).resolve()
    if a.media_out:
        OUT = Path(a.media_out).resolve()
    if a.cmd == "script":
        OUT.mkdir(parents=True, exist_ok=True)
        s = tour()
        SCRIPT.write_text(json.dumps(s, indent=1), encoding="utf-8")
        print(f"{SCRIPT}: {len(s['events'])} events, end state {s['end_state']}")
    elif a.cmd == "run":
        runs(Path(a.out or F / "logs" / "preview" / f"realtime_{dt.datetime.now():%Y%m%d-%H%M%S}"))
    else:
        r = video(a.run_dir, a.none, dump_ref=a.dump_ref, dump_only=a.dump_only)
        if a.dump_only:
            print(json.dumps(r, indent=1))
            return
        print(json.dumps({k: r[k] for k in ("outputs", "real_elapsed_s", "video_frames", "frames_presented",
                                             "frames_shown_in_video")}, indent=1))



if __name__ == "__main__":
    main()
