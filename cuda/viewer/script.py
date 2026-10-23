"""Manipulation scripts. Fixed before measuring: viewer/scripts/*.json.

Trajectory (start (x0s, y0s, epss)), 10 s, events spread evenly, gaps <= 50 ms, >= 60 events/s:
  0 - 5 s     drag only (eps = epss): the centre runs twice around the circle of radius r = 0.5*epss through the
              start point (circle centre (x0s - r, y0s)) and comes back to the start. 64 events/s = 320 drags.
              Each drag is an integer number of render-target pixels: the target offset of the circle point is
              rounded to whole pixels and the drag is the difference of consecutive rounded offsets, so the
              drags add up to zero and the end is the start again up to float rounding.
  5 - 7.5 s   wheel zoom-in only: 22 notches as 154 events of 1/7 notch (61.6 events/s); eps -> epss*0.9^22 < epss/10
  7.5 - 10 s  wheel zoom-out only: the same 154 events of -1/7 notch.

usage: python viewer/script.py   (writes four *_manip / *_scroll scripts and two *_one_notch scripts; legacy files unchanged)
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import params  # noqa: E402

SCENES = {"D1": (0.0, 0.0, 50.0), "Z1": (-0.43138, 0.895038, 0.1)}
DRAG_EVENTS, DRAG_SECONDS = 320, 5.0
ZOOM_EVENTS, ZOOM_NOTCHES, ZOOM_SECONDS = 154, 22, 2.5


def manipulation(start):
    x0s, y0s, epss = start
    r = 0.5 * epss
    events = []
    # drag phase: target offset in render-target pixels of the circle point, rounded; drag = difference
    prev = (0, 0)
    for i in range(1, DRAG_EVENTS + 1):
        th = 4 * math.pi * i / DRAG_EVENTS
        px, py = (x0s - r) + r * math.cos(th), y0s + r * math.sin(th)
        # x0 <- x0 - dx*(2eps/1919)  ->  dx = -(px - x0s) * 1919 / (2 eps);  y0 <- y0 + dy*(2eps_y/1079)
        X = round(-(px - x0s) * 1919 / (2 * epss))
        Y = round((py - y0s) * 1079 / (2 * params.eps_y(epss)))
        if i == DRAG_EVENTS:
            X, Y = 0, 0
        events.append({"t": DRAG_SECONDS * i / DRAG_EVENTS, "type": "drag", "dx": X - prev[0], "dy": Y - prev[1]})
        prev = (X, Y)
    step = ZOOM_NOTCHES / ZOOM_EVENTS  # 1/7 notch
    for i in range(1, ZOOM_EVENTS + 1):
        events.append({"t": DRAG_SECONDS + ZOOM_SECONDS * i / ZOOM_EVENTS, "type": "wheel", "steps": step})
    for i in range(1, ZOOM_EVENTS + 1):
        events.append({"t": DRAG_SECONDS + ZOOM_SECONDS + ZOOM_SECONDS * i / ZOOM_EVENTS, "type": "wheel",
                       "steps": -step})
    for k, ev in enumerate(events):
        ev["i"] = k
    return events


def scroll(start):
    """Ten seconds, 600 uniformly spaced wheel-in events of 1/6 notch."""
    events = [{"i": i - 1, "t": 10.0 * i / 600, "type": "wheel", "steps": 1 / 6}
              for i in range(1, 601)]
    assert params.replay(start, events)[2] >= 1e-13
    return events


def one_notch(start):
    """One second rest, one notch, then one second rest (duration is stored separately)."""
    return [{"i": 0, "t": 1.0, "type": "wheel", "steps": 1.0}]


def write_deepzoom():
    """Generate only the new artifacts; leave every existing legacy script untouched."""
    out = HERE / "scripts"
    scenes = json.loads((out / "deepzoom_scenes.json").read_text(encoding="utf-8"))["scenes"]
    files = {}
    for scene, spec in scenes.items():
        for kind, generate in (("manip", manipulation), ("scroll", scroll)):
            name = f"{scene}_{kind}"
            files[name] = {"scene": scene, "script_kind": kind, "start": spec["start"],
                           "duration": 10.0, "events": generate(spec["start"])}
        if scene in ("D1", "DZb"):
            files[f"{scene}_one_notch"] = {"scene": scene, "script_kind": "one_notch", "start": spec["start"],
                                           "duration": 2.0, "events": one_notch(spec["start"])}
    for name, data in files.items():
        (out / f"{name}.json").write_text(json.dumps(data, indent=0) + "\n", encoding="utf-8")
    return files


def short(start, moves, pause_after=None):
    """A short test script: listed events 30 ms apart; optional extra event after a pause (fast-mode return)."""
    events = [dict(ev, t=0.03 * (k + 1)) for k, ev in enumerate(moves)]
    if pause_after is not None:
        events.append(dict(pause_after[1], t=events[-1]["t"] + pause_after[0]))
    for k, ev in enumerate(events):
        ev["i"] = k
    return events


def write_all():
    out = HERE / "scripts"
    out.mkdir(exist_ok=True)
    files = {}
    for name, start in SCENES.items():
        files[name] = {"scene": name, "start": list(start), "events": manipulation(start)}
    # other positions for the precise-frame equivalence test (+ one extra input after the precise frame)
    files["P_left_zoom"] = {"scene": "D1", "start": [0.0, 0.0, 5.0],
                            "events": short(None, [{"type": "drag", "dx": 300, "dy": -120}, {"type": "wheel", "steps": 3}],
                                            pause_after=(2.5, {"type": "drag", "dx": -40, "dy": 25}))}
    files["P_deep"] = {"scene": "Z1", "start": [-0.43138, 0.895038, 0.1],
                       "events": short(None, [{"type": "wheel", "steps": 1}] * 8 + [{"type": "drag", "dx": -77, "dy": 51}],
                                       pause_after=(2.5, {"type": "wheel", "steps": -1}))}
    files["P_negreal"] = {"scene": "Z2", "start": [-2.0, 0.0, 1.0],
                          "events": short(None, [{"type": "drag", "dx": 13, "dy": 0}, {"type": "wheel", "steps": -0.5}],
                                          pause_after=(2.5, {"type": "drag", "dx": 0, "dy": -9}))}
    # mouse-callback test: several moves while the button stays held (each move must be measured from the previous
    # one: window 1 px + 1 px = render target dx [2, 2], not [2, 4]), a release + move, a new press, more moves
    d = lambda dx, dy: {"type": "drag", "dx": dx, "dy": dy}  # noqa: E731
    files["P_drag_multi"] = {"scene": "D1", "start": [0.0, 0.0, 5.0],
                             "events": short(None, [d(2, 0), d(2, 0), d(-6, 4), d(0, -2), {"type": "wheel", "steps": 1},
                                                    d(4, 4), d(4, -2), d(2, 2), {"type": "wheel", "steps": -1}],
                                             pause_after=(2.5, d(2, 2)))}
    for name, data in files.items():
        if name in SCENES:
            data["spec"] = __doc__.split("Trajectory")[1].split("usage:")[0].strip()
        (out / f"{name}.json").write_text(json.dumps(data, indent=0), encoding="utf-8")
    return files


def check(data):
    """Script rules: >= 60 events/s, gaps <= 50 ms, phases, and the end parameters = start up to rounding."""
    ev = data["events"]
    ts = [0.0] + [e["t"] for e in ev]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    end = params.replay(data["start"], ev)
    x0s, y0s, epss = data["start"]
    drags = [e for e in ev if e["type"] == "drag"]
    return {"events": len(ev), "seconds": ev[-1]["t"], "rate": len(ev) / ev[-1]["t"], "max_gap": max(gaps),
            "drag_pixels": sum(math.hypot(e["dx"], e["dy"]) for e in drags), "end": end,
            "end_ok": abs(end[0] - x0s) <= 1e-12 * epss and abs(end[1] - y0s) <= 1e-12 * epss
            and abs(end[2] / epss - 1) <= 1e-12,
            "min_eps": min(params.replay(data["start"], ev, k)[2] for k in range(len(ev)))}


if __name__ == "__main__":
    for name, data in write_deepzoom().items():
        if name in SCENES:
            print(name, check(data))
