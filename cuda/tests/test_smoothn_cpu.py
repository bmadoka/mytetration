"""No window, no GPU: the smooth max_iter rule and the HUD depth field."""
import math

import glfw
import numpy as np
import pytest

from deep import hp, limits
from viewer import app, deep_state, params, presets

POWERS = {k: 500 + 20 * (k - 13) for k in range(13, 41)}  # the old per-decade value at every power of ten


def old_rule(eps):
    """The earlier staircase rule: +20 just below each power of ten."""
    return 500 if eps >= 1e-13 else 500 + 20 * (hp.depth_digits(eps) - 13)


def notch_walk(eps):
    """eps, eps x 0.9, ... down to the floor L, exactly as the viewer's wheel does it."""
    out = [eps]
    while params.apply_wheel(0.0, 0.0, out[-1], 1)[2] >= limits.L:
        out.append(params.apply_wheel(0.0, 0.0, out[-1], 1)[2])
    return out


def test_powers_of_ten_keep_the_old_values():
    for k, n in POWERS.items():
        eps = float(f'1e-{k}')
        assert limits.max_iter_for(eps) == n == old_rule(eps), eps
    assert [limits.max_iter_for(e) for e in (1e-15, 1e-20, 1e-30, 1e-40)] == [540, 640, 840, 1040]


def test_shallow_is_500():
    for eps in (5.0, 1.0, 1e-5, 1e-12, 5e-13, 1.0000000000000002e-13, 1e-13):
        assert limits.max_iter_for(eps) == 500 == params.MAX_ITER


def test_just_below_1e13_starts_at_500():
    assert limits.max_iter_for(math.nextafter(1e-13, 0)) == 500
    assert limits.max_iter_for(1e-13 * (1 - 1e-4)) == 500       # the renderer-switch eps
    assert limits.max_iter_for(9.99e-14) == 500                 # the old staircase said 520 here


def test_monotone_dense_sweep():
    eps = np.logspace(-13, -40, 400_001)
    eps[0] = 1e-13
    n = np.array([limits.max_iter_for(float(e)) for e in eps])
    assert np.all(np.diff(n) >= 0)              # eps decreasing -> max_iter never decreases
    assert np.max(np.diff(n)) <= 1
    assert n[0] == 500 and n[-1] == 1040


@pytest.mark.parametrize('start', [5.0, 1e-13, math.nextafter(1e-13, 1), 1.0000001e-13, 1.05e-13, 1.1111e-14,
                                   1.00001e-15, 1e-15, 3.7e-17, 1.2345e-20, 1e-30])
def test_one_notch_adds_at_most_one(start):
    walk = notch_walk(start)
    n = [limits.max_iter_for(e) for e in walk]
    steps = np.diff(n)
    assert steps.min() >= 0 and steps.max() <= 1   # bound: <= 2 per notch, including 1e-13 and 10x edges
    deep = [e for e in walk if e < 1e-13]
    if len(deep) > 30:
        # 20 per decade on average: one notch is 20 * log10(1 / 0.9) = 0.915 iterations
        assert abs((n[-1] - limits.max_iter_for(deep[0])) / (len(deep) - 1) - 20 * math.log10(1 / 0.9)) < 0.02


def test_every_decade_edge_from_both_sides():
    for k in range(13, 40):
        p = float(f'1e-{k}')
        above, below = math.nextafter(p, 1), math.nextafter(p, 0)
        assert limits.max_iter_for(below) - limits.max_iter_for(above) <= 1, p
        assert limits.max_iter_for(p * 0.9) - limits.max_iter_for(p) <= 1, p
        assert limits.max_iter_for(p / 0.9) <= limits.max_iter_for(p), p
        if k > 13:
            assert old_rule(p * (1 - 1e-6)) - old_rule(p) == 20   # control: the old rule jumped here
            assert limits.max_iter_for(p * (1 - 1e-6)) - limits.max_iter_for(p) == 0


def test_rule_never_exceeds_the_old_staircase():
    for e in np.logspace(-13.0001, -40, 20_001):
        assert limits.max_iter_for(float(e)) <= old_rule(float(e))


def test_invalid_eps_still_rejected():
    for bad in (0.0, -1e-20, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            limits.max_iter_for(bad)


class HudDisplay:
    def __init__(self):
        self.hud_text, self.calls = None, []

    def set_hud(self, text):
        self.hud_text = text
        self.calls.append(text)


def hud_viewer(start=(0.0, 0.0, 5.0), start_hp=None):
    v = app.Viewer.__new__(app.Viewer)
    v.glfw = glfw
    v.state, v.hp = deep_state.initial_states(start, start_hp)
    v.display = v.prev_state = v.state
    v.applied = v.drawn = -1
    v.last_input_t = v.first_new_input_t = v.limit_until = None
    v.input_t, v.input_log, v.frames = {}, [], []
    v.preset_active, v.preset_moved, v.hud = None, False, True
    v.disp = HudDisplay()
    v._update_hud()
    return v


def depth_of(text):
    head, eps_s, n_s = text.rsplit(' · ', 2)
    assert eps_s.startswith('eps ') and n_s.startswith('N ')
    return head, float(eps_s[4:]), int(n_s[2:])


def test_depth_text_format():
    assert presets.depth_text(3.2e-17) == f'eps 3.2e-17 · N {limits.max_iter_for(3.2e-17)}'
    assert presets.depth_text(5.0) == 'eps 5.0e+00 · N 500'
    assert presets.depth_text(1e-40) == 'eps 1.0e-40 · N 1040'
    assert presets.hud_text('m37', True, 1e-15) == 'm37 · 이동함 · eps 1.0e-15 · N 540'
    assert presets.hud_text(None, False, 1e-15) == presets.GUIDE + ' · eps 1.0e-15 · N 540'


def test_hud_shows_eps_and_n_on_every_input_shallow_and_deep():
    v = hud_viewer()
    assert v.disp.calls[0] == presets.GUIDE + ' · eps 5.0e+00 · N 500'
    seen = 0
    for k in range(300):  # 5 -> about 1e-13 (shallow), then on into the deep range
        v.on_input({'type': 'wheel', 'steps': 1})
        if k % 7 == 0:
            v.on_input({'type': 'drag', 'dx': 3, 'dy': -2})
        head, eps, n = depth_of(v.disp.hud_text)
        assert head == presets.GUIDE
        assert n == limits.max_iter_for(v.state[2])
        assert eps == float(f'{v.state[2]:.1e}')
        seen += v.state[2] < 1e-13
    assert seen > 0 and v.state[2] < 1e-13 < 5.0
    press = lambda code: v._on_key(None, code, 0, glfw.PRESS, 0)  # noqa: E731
    press(glfw.KEY_2)
    assert depth_of(v.disp.hud_text) == ('m191', 1e-15, 540)
    v.on_input({'type': 'drag', 'dx': 5, 'dy': 0})
    assert depth_of(v.disp.hud_text) == ('m191 · 이동함', 1e-15, 540)
    for _ in range(25):
        v.on_input({'type': 'wheel', 'steps': 1})
    assert depth_of(v.disp.hud_text) == ('m191 · 이동함', float(f'{v.state[2]:.1e}'), limits.max_iter_for(v.state[2]))


def test_hud_off_for_scripts_unless_asked():
    v = hud_viewer()
    v.hud = False
    v.disp = HudDisplay()
    v.on_input({'type': 'wheel', 'steps': 1})
    assert v.disp.calls == []
