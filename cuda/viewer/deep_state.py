"""Independent float/Decimal input paths, with subpixel drift resynchronization."""
import math
from deep import hp, limits
from viewer import params


def initial_states(start, start_hp=None):
    state = tuple(float(v) for v in start)
    if start_hp is None:
        high = hp.from_floats(*state)
    else:
        state = (float(start_hp[0]), float(start_hp[1]), state[2])
        high = hp.from_strings(*start_hp, state[2])
    return state, high


def wheel_clamped(state, ev):
    return ev['type'] == 'wheel' and params.apply_event(state, ev)[2] < limits.L


def apply_event(state, high, ev, record=None):
    """Return float state, HP state, clamped. A rejected wheel changes neither."""
    if record is not None:
        record['resync'] = False
    if wheel_clamped(state, ev):
        return state, high, True
    updated = params.apply_event(state, ev)
    updated_hp = hp.apply_event(high, ev)
    diff = center_diff(updated, updated_hp)
    if any(d.copy_abs() > hp.D('0.5') for d in diff):
        if record is not None:
            record.update(resync=True, pre_resync_center_diff_px=[str(d) for d in diff])
        updated = project(updated_hp)
    return updated, updated_hp, False


def project(high):
    return float(high[0]), float(high[1]), high[2]


def center_diff(state, high, eps=None):
    return [hp.CTX.divide(hp.CTX.subtract(hp.D(f), d), s)
            for f, d, s in zip(state[:2], high[:2], hp.spacing(state[2] if eps is None else eps))]


def diagnostics(state, high, eps=None):
    """Signed centre error and half-ulp limits, using the displayed pixel spacing."""
    spacing = hp.spacing(state[2] if eps is None else eps)
    diff = list(map(float, center_diff(state, high, eps)))
    bound = [float(hp.CTX.divide(hp.CTX.divide(hp.D(math.ulp(f)), hp.D(2)), s))
             for f, s in zip(state[:2], spacing)]
    return dict(center_diff_px=diff, f64_round_limit_px=bound,
                hp_center=[str(v) for v in high[:2]])
