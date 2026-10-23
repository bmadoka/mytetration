"""Recording from the viewer's own render target: a timed replay in media mode (render-target snapshots every
~0.1 s and at precise frames) -> preview_media/<scene>/preview.gif (quarter size, 480x270), during_manipulation.png
and after_precise.png (full 1920x1080 render target). Not a timed measurement (read-backs slow the frames).

usage: python viewer/make_media.py [D1 Z1]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(HERE))
import launch  # noqa: E402

for scene in sys.argv[1:] or ["Z1", "D1"]:
    d = launch.run(F / "viewer" / "scripts" / f"{scene}.json", "media", F / "preview_media" / scene)
    print(scene, sorted(p.name for p in d.iterdir()))
