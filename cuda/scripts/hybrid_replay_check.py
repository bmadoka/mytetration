"""Replay one timed hybrid run through the protected launch.run window path."""
import argparse
import json
import sys
import time
from pathlib import Path

F = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(F / "viewer"), str(F / "tests"), str(F)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    import viewer_check as vc
    start = time.perf_counter()
    result = vc.check_replay(a.run_dir, a.out)
    print(json.dumps({"output": str(a.out or a.run_dir / "hybrid_replay"),
                      "frames": len(result["frames"]), "bad_blocks": result["condition5"]["bad_blocks"],
                      "bad_pixels": result["bad_pixels"], "elapsed_s": time.perf_counter() - start}))
    records = result["records"]
    moving = result["condition5"]["moving_frames"]  # an empty replay must not pass
    if not result["frames"] or result["expected_frames"] == 0 or moving == 0:
        print("replay covers no moving present (empty or initial-only recording)", file=sys.stderr)
        return 2
    return int(bool(result["condition5"]["bad_blocks"] or result["bad_pixels"]
                    or result["render_target_bad_pixels"] or result["bound_source_mismatch"]
                    or any(records.values()) or len(result["frames"]) != result["expected_frames"]))


if __name__ == "__main__":
    sys.exit(main())
