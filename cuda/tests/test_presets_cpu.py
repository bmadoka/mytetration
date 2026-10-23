"""Number-key presets: no window, no GPU, no OS input.

The registered key callback (Viewer._on_key) is called directly, with the real GLFW key codes.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import glfw
import numpy as np
import pytest

from deep import hp
from viewer import app, deep_state, deep_view, notice, presets

F = Path(__file__).resolve().parents[1]
ORDER = {'1': 'm37', '2': 'm191', '3': 'm59', '4': 'm60b'}
VIEWER = app.Viewer  # the real class (one test swaps app.Viewer to capture main's start)
PRESET_KEYS = {getattr(glfw, 'KEY_' + d) for d in '01234'} | {getattr(glfw, 'KEY_KP_' + d) for d in '01234'}


def bits(state):
    return [float(v).hex() for v in state]


class HudDisplay:
    """Records the HUD lines the viewer hands to the display."""
    def __init__(self):
        self.hud_text = None
        self.calls = []

    def set_hud(self, text):
        self.hud_text = text
        self.calls.append(text)


def key_viewer(start=(0.0, 0.0, 5.0), start_hp=None):
    v = VIEWER.__new__(VIEWER)
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


def snapshot(v):
    return (bits(v.state), str(v.hp[0]), str(v.hp[1]), v.hp[2].hex(), v.applied, len(v.input_log),
            v.preset_active, v.preset_moved, v.disp.hud_text)


def press(v, code, action=None):
    v._on_key(None, code, 0, glfw.PRESS if action is None else action, 0)


def test_preset_file_is_tracked_values_of_locations_json():
    data = json.loads((F / 'viewer/presets.json').read_text(encoding='utf-8'))
    rows = json.loads((F / 'tests/data/deepzoom2_positions.json').read_text(encoding='utf-8'))['positions']
    loc = {r['id']: r for r in rows}
    assert [(p['key'], p['name']) for p in data['presets']] == list(ORDER.items())
    for p in data['presets']:
        # 80-digit centre strings copied verbatim; the viewer reads only viewer/presets.json
        assert (p['x0'], p['y0']) == (loc[p['name']]['x0'], loc[p['name']]['y0'])
        assert float.fromhex(p['eps_hex']) == float(p['eps']) == 1e-15
    assert data['start'] == {'key': '0', 'name': None, 'x0': '0', 'y0': '0', 'eps': '5', 'eps_hex': (5.0).hex()}
    assert 'logs' not in Path(presets.FILE).relative_to(F).parts


@pytest.mark.parametrize('digit', '1234')
@pytest.mark.parametrize('row', ['KEY_', 'KEY_KP_'])
def test_digit_key_sets_preset_bit_for_bit(digit, row):
    entry = json.loads((F / 'viewer/presets.json').read_text(encoding='utf-8'))['presets'][int(digit) - 1]
    v = key_viewer()
    v.on_input({'type': 'drag', 'dx': 37, 'dy': -11})
    v.on_input({'type': 'wheel', 'steps': 3})
    press(v, getattr(glfw, row + digit))
    assert v.applied == 2 and v.input_log[-1]['type'] == 'key' and v.input_log[-1]['key'] == digit
    assert v.input_log[-1]['preset'] == ORDER[digit]
    assert (str(v.hp[0]), str(v.hp[1])) == (entry['x0'], entry['y0'])           # high-precision centre strings
    assert v.hp[2].hex() == v.state[2].hex() == entry['eps_hex']                  # eps bits
    assert bits(v.state[:2]) == [float(entry['x0']).hex(), float(entry['y0']).hex()]
    assert v.hp == hp.from_strings(entry['x0'], entry['y0'], float.fromhex(entry['eps_hex']))
    # same request key the deep worker gets: the centre strings are unchanged by state_key normalisation
    assert deep_view.state_key(v.hp) == (entry['x0'], entry['y0'], entry['eps_hex'])
    assert v.disp.hud_text == ORDER[digit] + ' · eps 1.0e-15 · N 540'  # name field + depth field (eps, N)
    assert v.snap_display is True


def test_wheel_keeps_preset_centre_and_ten_notches():
    v = key_viewer()
    press(v, glfw.KEY_1)
    centre = (str(v.hp[0]), str(v.hp[1]))
    eps = v.state[2]
    for _ in range(10):
        v.on_input({'type': 'wheel', 'steps': 1})
        eps *= 0.9
    assert (str(v.hp[0]), str(v.hp[1])) == centre
    assert v.hp[2].hex() == v.state[2].hex() == eps.hex()
    assert v.disp.hud_text == 'm37 · eps 3.5e-16 · N 549'  # the wheel does not move the centre; depth follows eps
    assert v.disp.hud_text == 'm37' + presets.DEPTH_SEP + presets.depth_text(v.state[2])


@pytest.mark.parametrize('row', ['KEY_', 'KEY_KP_'])
def test_key_zero_returns_to_the_start_of_main(monkeypatch, row):
    captured = {}

    class Capture:
        def __init__(self, start, mode, readback, log, **kw):
            captured.update(start=start, mode=mode, kw=kw)

        def run(self, events):
            captured['events'] = events

        def close(self):
            pass
    monkeypatch.setattr(app, 'preflight_interactive', lambda: None)
    monkeypatch.setattr(app, 'Viewer', Capture)
    assert app.main(['--interactive']) == 0
    assert captured['mode'] == 'interactive' and captured['kw'] == {} and captured['events'] is None
    main_state, main_hp = deep_state.initial_states(captured['start'], None)   # exactly how main's Viewer starts
    assert bits(main_state) == bits((0.0, 0.0, 5.0))
    v = key_viewer([0.0, 0.0, 1e-20], ['-1.0851625774713976413892441561578844795', '.1821715056672227'])
    press(v, glfw.KEY_3)
    v.on_input({'type': 'drag', 'dx': 5, 'dy': 9})
    press(v, getattr(glfw, row + '0'))
    assert bits(v.state) == bits(main_state) == bits((0.0, 0.0, 5.0))
    assert v.hp == main_hp and repr(v.hp) == repr(main_hp) and (str(v.hp[0]), str(v.hp[1])) == ('0', '0')
    assert v.input_log[-1]['preset'] == 'start'
    assert v.disp.hud_text == presets.GUIDE + ' · eps 5.0e+00 · N 500' and v.preset_active is None


def test_other_keys_and_release_repeat_change_nothing():
    v = key_viewer()
    press(v, glfw.KEY_2)
    v.on_input({'type': 'drag', 'dx': 3, 'dy': 4})
    before = snapshot(v)
    codes = sorted({getattr(glfw, n) for n in dir(glfw) if n.startswith('KEY_') and isinstance(getattr(glfw, n), int)})
    others = [c for c in codes if c not in PRESET_KEYS]
    assert glfw.KEY_5 in others and glfw.KEY_9 in others and glfw.KEY_KP_5 in others and glfw.KEY_A in others
    assert len(others) >= 100
    for code in others:
        for action in (glfw.PRESS, glfw.RELEASE, glfw.REPEAT):
            press(v, code, action)
    for code in PRESET_KEYS:  # digit keys: only PRESS counts; release and auto-repeat do nothing
        for action in (glfw.RELEASE, glfw.REPEAT):
            press(v, code, action)
    assert snapshot(v) == before


def test_hud_name_moved_marker_and_guide():
    v = key_viewer()
    assert presets.GUIDE == '1~4: 무늬 위치, 0: 처음으로'
    assert v.disp.calls[0] == '1~4: 무늬 위치, 0: 처음으로 · eps 5.0e+00 · N 500'  # guide + depth field
    v.on_input({'type': 'drag', 'dx': 10, 'dy': 0})
    assert v.disp.hud_text == '1~4: 무늬 위치, 0: 처음으로 · eps 5.0e+00 · N 500'  # no preset yet: the guide stays
    press(v, glfw.KEY_4)
    assert v.disp.hud_text == 'm60b · eps 1.0e-15 · N 540'
    v.on_input({'type': 'wheel', 'steps': -2})
    assert v.disp.hud_text == 'm60b · eps 1.2e-15 · N 538'
    v.on_input({'type': 'drag', 'dx': 0, 'dy': 1})
    assert v.disp.hud_text == 'm60b · 이동함 · eps 1.2e-15 · N 538' and v.preset_moved
    press(v, glfw.KEY_4)                          # the same key again: back on the preset, marker cleared
    assert v.disp.hud_text == 'm60b · eps 1.0e-15 · N 540' and not v.preset_moved
    press(v, glfw.KEY_0)
    assert v.disp.hud_text == '1~4: 무늬 위치, 0: 처음으로 · eps 5.0e+00 · N 500'


def test_hud_image_is_binary_top_right_and_clear_of_notices():
    notice_right = max(int(np.nonzero(notice.text_mask(t).any(axis=1))[0].max()) // 2
                       for t in (notice.LIMIT_TEXT, notice.range_text('near_zero'), notice.range_text('branch_cut'),
                                 notice.range_text('beyond_L')))
    for text in (presets.GUIDE, 'm37', 'm191 · 이동함', presets.hud_text(None, False, 5.0),
                 presets.hud_text('m191', True, 1.0628819999999999e-40)):  # longest lines with the depth field
        im = presets.hud_image(text)
        assert im.shape == (540, 960, 4) and set(np.unique(im[..., 3])) == {0, 255}
        opaque = im[..., 3] == 255
        ys, xs = np.nonzero(opaque)
        assert ys.min() >= 10 and ys.max() < 60 and xs.max() <= 950 and xs.min() > notice_right
        white = opaque & (im[..., 0] == 255)
        assert white.sum() > 50 and np.all(im[opaque & ~white][:, :3] == 0)


def test_preset_frame_jumps_without_smooth_zoom():
    """The first frame after a key shows the preset state itself (smooth zoom would stop between the depths)."""
    m37 = presets.table()['1']
    v = key_viewer([0.0, 0.0, 1e-20], [m37['x0'], m37['y0']])
    v.decision_t = 0.0
    shown = []
    v._deep_frame = lambda kind, p, t: shown.append(p)
    press(v, glfw.KEY_2)
    target = v.state
    assert app.params.smooth_step(v.display, target, 0.013)[2] != target[2]  # what a wheel jump would show
    v.frame('fast', 0.013)
    assert bits(shown[-1]) == bits(target)
    v.on_input({'type': 'wheel', 'steps': -40})
    v.frame('fast', 0.026)                        # later inputs zoom smoothly again
    assert bits(shown[-1]) != bits(v.state)


def test_script_key_goes_through_registered_key_callback():
    v = key_viewer()
    v.disp.win = None
    v.render = SimpleNamespace(SCALE=2)
    v._cursor = v._drag_from = None
    v.press_sources = []
    seen = []

    def key_cb(*args):
        seen.append(args)
        v._on_key(*args)
    cbs = dict(cursor=v._on_cursor, button=v._on_button, scroll=v._on_scroll, key=key_cb)
    st = {'pressed': False, 'pos': (100.0, 40.0), 'first_press_done': False}
    v.glfw = SimpleNamespace(**{n: getattr(glfw, n) for n in dir(glfw) if n.isupper()},
                             get_cursor_pos=lambda win: st['pos'])
    for ev in ({'type': 'drag', 'dx': 20, 'dy': 0}, {'type': 'key', 'key': '1'}, {'type': 'wheel', 'steps': 1},
               {'type': 'key', 'key': '0'}):
        v._deliver_via_callbacks(ev, cbs, st)
    assert [r['type'] for r in v.input_log] == ['drag', 'key', 'wheel', 'key']
    assert [a[3] for a in seen] == [glfw.PRESS, glfw.RELEASE, glfw.PRESS, glfw.RELEASE]
    assert not st['pressed']                     # a held button is released before the key, like the wheel
    assert bits(v.state) == bits((0.0, 0.0, 5.0))
