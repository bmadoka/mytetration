"""Compare zero-based MP4 frames with lossless composed RGB references (CPU only)."""
import argparse
import json
import math
import shutil
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def parse_frame_indices(value):
    """Comma-separated nonnegative frame numbers, deduplicated in input order."""
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token or not token.isascii() or not token.isdecimal() for token in tokens):
        raise argparse.ArgumentTypeError("frames must be comma-separated nonnegative integers")
    return list(dict.fromkeys(int(token) for token in tokens))


def luma_psnr(reference, decoded):
    reference, decoded = np.asarray(reference), np.asarray(decoded)
    if reference.shape != decoded.shape or reference.ndim != 3 or reference.shape[-1] != 3 or not reference.size:
        raise ValueError("reference and decoded frames must have the same nonempty H x W x 3 RGB shape")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    delta = reference.astype(np.float64) @ weights - decoded.astype(np.float64) @ weights
    mse = float(np.mean(delta ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def decode_frame(mp4, index, shape):
    result = subprocess.run([FFMPEG, "-v", "error", "-i", str(mp4), "-map", "0:v:0",
                             "-vf", f"select=eq(n\\,{index})", "-vsync", "0", "-frames:v", "1",
                             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                            capture_output=True, check=True)
    if len(result.stdout) != math.prod(shape):
        raise ValueError(f"{mp4}: frame {index} missing or RGB size differs from reference {shape}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(shape)


def binary_mismatch(reference, decoded, box=(10, 10, 222, 112)):
    """Fraction of different luma>127 bits outside the inclusive overlay rectangle."""
    reference, decoded = np.asarray(reference), np.asarray(decoded)
    if reference.shape != decoded.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("reference and decoded must have identical RGB shape")
    mask = np.ones(reference.shape[:2], bool)
    x0, y0, x1, y1 = box
    h, w = mask.shape
    mask[max(0,y0):min(h,y1+1), max(0,x0):min(w,x1+1)] = False
    weights = np.array([.299, .587, .114], dtype=np.float64)
    mismatch = ((reference.astype(np.float64) @ weights > 127)
                != (decoded.astype(np.float64) @ weights > 127))
    if not mask.any():
        raise ValueError("overlay leaves no comparison pixels")
    return float(mismatch[mask].mean())


def mismatch_summary(values):
    pairs = list(values.items())
    ratios = [v for _, v in pairs]
    return {"max": max(ratios), "p95": float(np.percentile(ratios, 95)),
            "median": float(np.median(ratios)),
            "frames_over_2_percent": [int(i) for i, v in pairs if v > .02],
            "frames_over_3_percent": [int(i) for i, v in pairs if v > .03]}


def decode_frames(mp4, shape=(540, 960, 3)):
    """One ffmpeg pass; yield every decoded RGB frame once, with bounded memory."""
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(mp4), "-map", "0:v:0", "-vsync", "0",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = math.prod(shape)
    try:
        while True:
            data = p.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                raise ValueError("partial decoded frame")
            yield np.frombuffer(data, np.uint8).reshape(shape)
        error = p.stderr.read().decode("utf-8", errors="replace")
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg decode failed: {error}")
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        p.stdout.close()
        p.stderr.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mp4", nargs="+", type=Path)
    ap.add_argument("--ref-dir", type=Path)
    ap.add_argument("--recording", type=Path, help="viewer recording to recompose with the encoder composition")
    ap.add_argument("--all-frames", action="store_true", help="decode every frame once; requires --recording")
    ap.add_argument("--frames", type=parse_frame_indices)
    ap.add_argument("--out", type=Path, help="JSON destination (default: <ref-dir>/video_psnr.json)")
    ap.add_argument("--compare", type=int, help="zero-based frame for a side-by-side PNG")
    ap.add_argument("--compare-out", type=Path)
    a = ap.parse_args(argv)
    if (a.compare is None) != (a.compare_out is None) or (a.compare is not None and a.compare < 0):
        ap.error("--compare must be nonnegative and supplied together with --compare-out")
    if a.all_frames and not a.recording:
        ap.error("--all-frames requires --recording")
    if not a.all_frames and (not a.frames or not a.ref_dir):
        ap.error("sampled mode requires --frames and --ref-dir")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
    import make_realtime_video as mv
    box = mv.OVERLAY_BOX
    indices = list(dict.fromkeys((a.frames or []) + ([] if a.compare is None else [a.compare])))
    references = {}
    if not a.all_frames:
        for index in indices:
            with Image.open(a.ref_dir / f"frame_{index}.png") as im:
                references[index] = np.asarray(im.convert("RGB"))
    scores, comparison = [], []
    counted = None
    for mp4 in a.mp4:
        values, binary = {}, {}
        if a.all_frames:
            pairs = enumerate(zip_longest(mv.iter_composed_frames(a.recording), decode_frames(mp4), fillvalue=None))
        else:
            pairs = ((i, (references[i], decode_frame(mp4, i, references[i].shape))) for i in indices)
        n = 0
        for index, (reference, decoded) in pairs:
            if reference is None or decoded is None:
                raise ValueError(f"{mp4}: decoded and recomposed frame counts differ at frame {index}")
            n += 1
            if a.all_frames or index in a.frames:
                score = luma_psnr(reference, decoded)
                values[str(index)] = "inf" if math.isinf(score) else score
                binary[str(index)] = binary_mismatch(reference, decoded, box)
            if index == a.compare:
                if not comparison:
                    comparison.append(reference)
                comparison.append(decoded)
        if not binary:
            raise ValueError("no frames compared")
        if counted is not None and n != counted:
            raise ValueError("videos have different frame counts")
        counted = n
        scores.append({"mp4": str(mp4), "psnr_db": values, "binary_mismatch": binary,
                       "binary_summary": mismatch_summary(binary), "frames_compared": len(binary)})
    table_indices = [int(i) for i in scores[0]["psnr_db"]]
    print("| frame | " + " | ".join(str(mp4).replace("|", r"\|") for mp4 in a.mp4) + " |")
    print("| --- | " + " | ".join("---:" for _ in a.mp4) + " |")
    for index in table_indices:
        values = [entry["psnr_db"][str(index)] for entry in scores]
        print(f"| {index} | " + " | ".join(v if isinstance(v, str) else f"{v:.4f}" for v in values) + " |")
    if a.compare is not None:
        Image.fromarray(np.concatenate(comparison, axis=1)).save(a.compare_out)
    result = {"ref_dir": str(a.ref_dir) if a.ref_dir else None, "frames": table_indices,
              "all_frames": a.all_frames, "recording": str(a.recording) if a.recording else None,
              "overlay_box_inclusive": list(box), "binary_threshold": 127, "luma": [0.299, 0.587, 0.114], "peak": 255,
              "videos": scores, "compare_frame": a.compare,
              "compare_out": str(a.compare_out) if a.compare_out is not None else None}
    destination = a.out or (a.ref_dir or a.recording) / "video_psnr.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
