"""Opt-in window test: MYTET_PRESETS_WINDOW=1. One protected window run of viewer/scripts/presets_window.json in callbacks
mode: key 1 -> 3 wheel notches -> 40 px drag -> key 0, keys through the registered GLFW key callback (no OS input).
Protection = the --script rules: launch.wait_for_room (idle >= 5 min, no full-screen app, GPU free, memory; re-checked by
the viewer's own preflight), window shown without activation. No CPU-contamination wait (launch.run's cpu_guard):
this test measures no time.

Checked on the screen read-backs (readback 'precise'):
- every final deep picture after key 1 is at the preset centre strings, its render target equals the worker's final
  map, and the window back buffer equals that picture down-scaled with exactly the HUD pixels of its line on top;
- after key 0 the precise frame is (0, 0, 5) and its render target AND window equal the first frame's, bit for bit.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

F = Path(__file__).resolve().parents[1]
SCRIPT = F / 'viewer/scripts/presets_window.json'
pytestmark = pytest.mark.skipif(os.environ.get('MYTET_PRESETS_WINDOW') != '1',
                                reason='window test: set MYTET_PRESETS_WINDOW=1 to run')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def unpack(bits):
    return np.unpackbits(np.asarray(bits, np.uint8), count=1920 * 1080).reshape(1920, 1080).astype(bool)


def render_target(m):
    rt = np.empty((1080, 1920, 4), np.uint8)
    rt[..., :3] = (m.T * 255)[..., None]
    rt[..., 3] = 255
    return rt


def window(rt, hud_text):
    """Window rule (pixel (u, v) = render target (2u, 2v), rows bottom -> top) with the binary HUD on top."""
    from viewer import presets
    bb = rt[::2, ::2, :3].copy()
    if hud_text:
        hud = presets.hud_image(hud_text)[::-1]
        on = hud[..., 3] == 255
        bb[on] = hud[on][:, :3]
    return bb


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_key1_wheel_drag_key0_on_screen():
    from viewer import launch, presets

    def name_field(text):  # the HUD line without its depth field
        return text.split(presets.DEPTH_SEP + 'eps ')[0]

    def hud(name, moved, eps_hex):
        return presets.hud_text(name, moved, float.fromhex(eps_hex))
    sys.path.insert(0, str(F / 'scripts'))
    from deepzoom2_eval import structure
    out = F / 'logs/presets/window' / f'{dt.datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}'
    pre = launch.wait_for_room(float(os.environ.get('MYTET_WINDOW_MAX_WAIT', '3600')))
    out.mkdir(parents=True, exist_ok=False)
    cp = subprocess.run([sys.executable, '-m', 'viewer', '--script', str(SCRIPT), '--mode', 'callbacks',
                         '--readback', 'precise', '--log', str(out)], cwd=str(F), capture_output=True, text=True,
                        env=dict(os.environ, MPLCONFIGDIR=str(F / '.mplconfig')), timeout=600)
    (out / 'launch.json').write_text(json.dumps(dict(pre_checks=pre, returncode=cp.returncode,
                                                     stderr_tail=cp.stderr[-3000:]), indent=1), encoding='utf-8')
    assert cp.returncode == 0, cp.stderr[-2000:]
    meta = json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    frames, inputs = jsonl(out / 'frames.jsonl'), jsonl(out / 'inputs.jsonl')
    script = json.loads(SCRIPT.read_text(encoding='utf-8'))
    assert meta['hud'] is True and meta['registered_key_callback'] is True
    assert meta['events_delivered'] == len(script['events']) == len(inputs)
    assert [r['type'] for r in inputs] == [e['type'] for e in script['events']]
    assert [r.get('preset') for r in inputs if r['type'] == 'key'] == ['m37', 'start']
    assert not meta['window_state_at_end']['foreground_is_viewer']

    m37 = presets.table()['1']
    eps = [m37['eps']]
    for _ in range(3):
        eps.append(eps[-1] * 0.9)
    finals = {}
    with np.load(out / 'deep_final_maps.npz') as archive:
        maps = {k: archive[k] for k in archive.files}
    summary = []
    for f in frames:
        if f.get('underlying_kind') != 'deep_final' or f.get('intermediate') is not False:
            continue
        assert (f['deep']['x0'], f['deep']['y0']) == (m37['x0'], m37['y0']) or name_field(f['hud']) == 'm37 · 이동함', f['seq']
        m = unpack(maps[str(f['source_request_id'])])
        rt = render_target(m)
        assert f['rt_sha256'] == sha(rt), f['seq']                      # the screen's picture is the final map
        assert f['bb_sha256'] == sha(window(rt, f['hud'])), f['seq']    # window = picture + exactly the HUD line
        finals.setdefault((f['deep']['eps_hex'], f['hud']), f)
        if f['deep']['eps_hex'] in (eps[0].hex(), eps[3].hex()) and f['hud'] == hud('m37', False, f['deep']['eps_hex']):
            pairs = structure(m)['boundary_pairs']
            assert pairs >= 27_500
            summary.append(dict(seq=f['seq'], eps_hex=f['deep']['eps_hex'], boundary_pairs=pairs))
    # HUD = name field + its depth field (eps and N of that state)
    assert (eps[0].hex(), hud('m37', False, eps[0].hex())) in finals  # key 1: the preset itself, name on the HUD
    assert (eps[3].hex(), hud('m37', False, eps[3].hex())) in finals  # 3 notches in: same centre, still the name
    assert (eps[3].hex(), hud('m37', True, eps[3].hex())) in finals   # after the drag: the moved marker
    assert finals[(eps[3].hex(), hud('m37', True, eps[3].hex()))]['deep']['x0'] != m37['x0']

    with np.load(out / 'precise_maps.npz') as archive:
        precise = {k: archive[k] for k in archive.files}
    first, last = frames[0], frames[-1]
    assert first['kind'] == 'initial' and last['kind'] == 'precise'
    start_hex = [(0.0).hex(), (0.0).hex(), (5.0).hex()]
    assert first['display_params_hex'] == last['display_params_hex'] == start_hex
    start_hud = presets.hud_text(None, False, 5.0)
    assert first['hud'] == last['hud'] == start_hud and name_field(start_hud) == presets.GUIDE
    for f in (first, last):
        rt = render_target(unpack(precise[f['precise_id']]))
        assert f['rt_sha256'] == sha(rt) and f['bb_sha256'] == sha(window(rt, start_hud))
    assert last['rt_sha256'] == first['rt_sha256'] and last['bb_sha256'] == first['bb_sha256']
    (out / 'presets_window_check.json').write_text(json.dumps(dict(
        passed=True, deep_final_states=sorted(finals), boundary_pairs=summary,
        first_last_rt_sha=[first['rt_sha256'], last['rt_sha256']],
        first_last_bb_sha=[first['bb_sha256'], last['bb_sha256']]), indent=1, ensure_ascii=False), encoding='utf-8')
