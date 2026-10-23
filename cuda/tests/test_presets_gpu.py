"""Chosen GPU only: the four number-key presets show pattern at once and keep it while zooming in.

For each preset the final 1920 x 1080 deep picture is computed exactly as the viewer's deep worker requests it
(state_key centre strings, eps bits, max_iter = limits.max_iter_for(eps)) at the preset state and after 10 wheel
notches in (10 single-notch events through deep_state.apply_event, as the viewer applies them). Both must have
>= 27,500 boundary pairs (scripts/deepzoom2_eval.structure) and stay in range.
"""
import json
from pathlib import Path
import sys

import pytest

from deep import limits, perturb
from viewer import deep_state, deep_view, presets

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F / 'scripts'))
from perturb_tools import audit_gpu  # noqa: E402
from deepzoom2_eval import structure  # noqa: E402

MIN_PAIRS = 27_500


@pytest.fixture(scope='module', autouse=True)
def audit_engine_context(gpu, run_dir):
    gpu.cp.empty(1)
    audit_gpu('presets tests CUDA context', run_dir)


def worker_map(high):
    x0, y0, eps_hex = deep_view.state_key(high)
    eps = float.fromhex(eps_hex)
    return perturb.deep_map(x0, y0, eps, max_iter=limits.max_iter_for(eps))


@pytest.mark.parametrize('key', '1234')
def test_preset_and_ten_notches_in_have_pattern(key, run_dir):
    state, high = presets.states(key)
    rows = []
    for notches in (0, 10):
        s, h = state, high
        for _ in range(notches):
            s, h, clamped = deep_state.apply_event(s, h, {'type': 'wheel', 'steps': 1})
            assert not clamped
        assert (str(h[0]), str(h[1])) == (str(high[0]), str(high[1]))  # the wheel keeps the preset centre
        m, info = worker_map(h)
        metric = structure(m)
        rows.append(dict(key=key, name=presets.name(key), notches=notches, eps=h[2], eps_hex=h[2].hex(),
                         max_iter=limits.max_iter_for(h[2]), out_of_range=info['out_of_range'],
                         reason=info['reason'], boundary_pairs=metric['boundary_pairs'],
                         tile_ratio=metric['tile_ratio'], t_ms=info['t_ms'].get('total')))
        with (run_dir / 'presets_boundary_pairs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(rows[-1]) + '\n')
    for row in rows:
        assert not row['out_of_range'], row
        assert row['boundary_pairs'] >= MIN_PAIRS, row
