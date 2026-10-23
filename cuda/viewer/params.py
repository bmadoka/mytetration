"""The one place for the view state and the move / zoom formulas.

Render target = 1920 x 1080 grid of the original 16by9_H rule (n = 1920, eps_y = eps * (9/16), ny = 1080).
State = (x0, y0, eps); max_iter 500 and escape_radius 1e10 are fixed.

Drag by (dx, dy) render-target pixels (dx > 0 = cursor moved right, dy > 0 = cursor moved down; the picture's +y is
up) — the picture follows the cursor:
    x0 <- x0 - dx * (2 * eps / 1919)
    y0 <- y0 + dy * (2 * eps_y / 1079),   eps_y = eps * (9 / 16)
Wheel by s notches (s > 0 zoom in, s < 0 zoom out; s may be fractional, e.g. a precision wheel), fixed point = the
centre of the window (so x0, y0 do not change):
    s > 0:  eps <- eps * (0.9 ** s)          s < 0:  eps <- eps * ((1 / 0.9) ** (-s))
One notch in = x 0.9, one notch out = x 1/0.9. Every float operation is written exactly as above. Tests compute the
expected parameters from the event list with the contract equations re-encoded in tests/viewer_check.contract_step,
and pin these functions to them bit for bit (tests/test_viewer.py::test_formulas).
"""

import math

NX, NY = 1920, 1080
VARIANT = "16by9_H"
MAX_ITER = 500
ESCAPE_RADIUS = 1e10
K_IN = 0.9
K_OUT = 1 / 0.9


def eps_y(eps):
    return eps * (9 / 16)


def apply_drag(x0, y0, eps, dx, dy):
    return x0 - dx * (2 * eps / 1919), y0 + dy * (2 * eps_y(eps) / 1079), eps


def apply_wheel(x0, y0, eps, steps):
    if steps > 0:
        return x0, y0, eps * (K_IN ** steps)
    if steps < 0:
        return x0, y0, eps * (K_OUT ** (-steps))
    return x0, y0, eps


def window_to_render_target(dx_window, dy_window, scale):
    """A mouse move of (dx, dy) window pixels is (dx*scale, dy*scale) render-target pixels (window = render target
    shrunk by `scale`)."""
    return dx_window * scale, dy_window * scale


def apply_event(state, ev):
    """state = (x0, y0, eps); ev = {"type": "drag", "dx", "dy"} or {"type": "wheel", "steps"}."""
    x0, y0, eps = state
    if ev["type"] == "drag":
        return apply_drag(x0, y0, eps, ev["dx"], ev["dy"])
    if ev["type"] == "wheel":
        return apply_wheel(x0, y0, eps, ev["steps"])
    raise ValueError(ev["type"])


def replay(start, events, upto=None):
    """Parameters after applying events[0 .. upto] (inclusive; None = all) to start, with the formulas above."""
    state = tuple(start)
    for k, ev in enumerate(events):
        if upto is not None and k > upto:
            break
        state = apply_event(state, ev)
    return state


TAU = 0.06
SNAP = 0.02 * math.log(1 / 0.9)


def smooth_step(display, target, dt):
    """README smooth zoom S: drag immediately, capped exponential following in log(eps)."""
    if float(display[2]).hex() == float(target[2]).hex():
        return tuple(target)
    lt = math.log(target[2])
    ld = math.log(display[2])
    a = min(0.5, 1 - math.exp(-dt / TAU))
    ln_new = ld + a * (lt - ld)
    eps = target[2] if abs(lt - ln_new) <= SNAP else math.exp(ln_new)
    return target[0], target[1], eps
