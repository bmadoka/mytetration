"""CPU-only contract tests; never import the CUDA host module or CuPy."""
from decimal import Decimal, localcontext
import ast
import math
import sys
from pathlib import Path
import uuid

import mpmath as mp
import numpy as np
import pytest

from deep import hp, limits, cfn, reference, truth
from viewer import params
from perturb_cases import INPUTS, assert_components, oracle


def test_hp_drag_wheel_context():
    state = hp.from_strings('1.23456789', '-0.13579', 1e-40)
    e = Decimal(state[2])
    with localcontext(hp.CTX) as ctx:
        sx = 2*e/Decimal(1919)
        sy = 2*(e*9/16)/Decimal(1079)
        expected = (state[0]-Decimal(100)*sx, state[1]+Decimal(-17)*sy, state[2])
    with localcontext() as ctx:
        ctx.prec = 3
        moved = hp.apply_drag(state, 100, -17)
        assert moved == expected
        assert hp.apply_drag(hp.apply_drag(state, 100, 0), -100, 0) == state
    for steps in [1, -1, .25, -4.5, 0, 30]:
        state = hp.apply_wheel(state, steps)
        expected_eps = params.apply_wheel(0, 0, expected[2], steps)[2]
        assert state[2].hex() == expected_eps.hex()
        expected = (*expected[:2], expected_eps)
    assert hp.D(.1) == Decimal(.1)
    assert hp.D(state[0]) is state[0]
    assert hp.to_record(state) == [str(state[0]), str(state[1]), state[2], state[2].hex()]
    assert hp.apply_event(state, {'type':'drag', 'dx':1, 'dy':2}) == hp.apply_drag(state, 1, 2)
    assert hp.apply_event(state, {'type':'wheel', 'steps':2}) == hp.apply_wheel(state, 2)


def test_grid_endpoints_and_offsets():
    x, y, e = hp.from_strings('1.234567890123456789', '.34567890123456789', 1e-40)
    ex, ey = hp.extents(e)
    with localcontext() as ctx:
        ctx.prec = 2
        assert hp.grid_x(x, e, 0) == hp.CTX.subtract(x, ex)
        assert hp.grid_x(x, e, 1919) == hp.CTX.add(x, ex)
        assert hp.grid_y(y, e, 0) == hp.CTX.subtract(y, ey)
        assert hp.grid_y(y, e, 1079) == hp.CTX.add(y, ey)
    ref = reference.at_pixel(x, y, e, (960,540), 1, 70)
    gx, gy = reference.offsets(x, y, e, ref)
    assert len(np.unique(gx)) == 1920 and len(np.unique(gy)) == 1080
    assert gx[960] == gy[540] == 0
    assert ref.center == (str(hp.grid_x(x,e,960)), str(hp.grid_y(y,e,540)))


@pytest.mark.parametrize('name', ['cexpm1', 'clog1p'])
def test_complex_functions(name):
    func = getattr(cfn, name)
    array = func(INPUTS)
    for z, got in zip(INPUTS, array):
        assert_components(got, z, name)
        assert func(z) == got


def test_numpy_log1p_control_loses_real_part():
    z = complex(1e-45, 1e-45)
    target = oracle(z, 'clog1p').real
    assert abs(np.log1p(np.complex128(z)).real-target) > 4e-15*abs(target)
    assert abs(cfn.clog1p(z).real-target) <= 4e-15*abs(target)


def test_truth_known_stops_and_precision():
    assert len(truth.SAMPLE) == len(set(truth.SAMPLE)) == 625
    assert truth.SAMPLE[0] == (0,0) and truth.SAMPLE[-1] == (1919,1079)
    assert truth.verdict('2','0',10,50) == (True,4)
    assert truth.verdict('1','0',20,50) == (False,20)
    assert truth.verdict('0','0',20,50) == (False,20)
    for x,y in [('2','0'), ('1','0'), ('.2','.3'), ('-.3','.8')]:
        assert truth.verdict(x,y,30,50) == truth.verdict(x,y,30,80)
    points = truth.sample_points('1','.2',1e-40)
    assert len(set(points)) == 625
    assert points[0][0] == str(hp.grid_x('1',1e-40,0))


def test_truth_cache_and_spawn():
    tmp_path = Path(__file__).resolve().parents[1]/'logs/perturb/cpu_test_cache'/uuid.uuid4().hex
    a, stop = truth.truth_samples('1','0',1e-40,1,60,processes=1,cache_dir=tmp_path)
    b, stops2 = truth.truth_samples('1','0',1e-40,1,60,processes=2,cache_dir=tmp_path)
    assert np.array_equal(a,b) and np.array_equal(stop,stops2)
    # A separate key really uses Windows spawn rather than the cached result.
    c, _ = truth.truth_samples('1','0',1e-40,0,60,processes=2)
    assert not c.any()


def test_reference_direct_logs_and_initial():
    ref = reference.orbit('2','0',20,70)
    assert list(ref.z[:3]) == [0j, 1+0j, 2+0j]
    assert ref.n_ref == 4 and len(ref.z) == len(ref.log_abs) == 7
    with mp.workdps(70):
        z, lc = mp.mpc(2), mp.log(2)
        for i in range(5):
            assert ref.z[i+2] == complex(z)
            assert ref.log_abs[i+2] == float(mp.log(abs(z)))
            if i<4:
                z = mp.exp(z*lc)
    assert reference.range_issue(ref,20) is None
    quiet = reference.orbit('1','0',5,50)
    assert quiet.n_ref is None and len(quiet.z)==8
    assert quiet.z[0]==0 and np.all(quiet.z[1:]==1)
    assert quiet.log_abs[0]==-math.inf and np.all(quiet.log_abs[1:]==0)


def test_reference_exponent_injection():
    ref = reference.orbit('1','0',3,50)
    ref.log_abs[3] = -1001*math.log(2)
    assert reference.range_issue(ref,3) is None
    ref.log_abs[3] = 0
    ref.z[3] = 0
    assert reference.range_issue(ref,3) is None
    ref.z[3] = complex(np.nextafter(0.,1.),0)
    assert reference.range_issue(ref,3) is None
    ref.z[3] = complex(math.inf,0)
    assert reference.range_issue(ref,3)['n']==1
    # Escaping last values may round to inf while their log remains finite.
    ref.n_ref = 1
    assert reference.range_issue(ref,1) is None
    ref.z[3] = complex(math.nan,0)
    assert reference.range_issue(ref,1)['n']==1
    ref.z[3] = 1
    ref.log_abs[3] = math.nan
    assert reference.range_issue(ref,1)['n']==1
    tiny = reference.orbit('1e-400','0',1,70)
    assert tiny.z[2] == 0 and math.isfinite(tiny.log_abs[2])
    assert reference.range_issue(tiny,1) is None


def test_limits_and_no_gpu_import():
    assert limits.view_reason('1','1',1e-41)=='beyond_L'
    assert limits.view_reason('0','0',1e-20)=='near_zero'
    assert limits.view_reason('-1','0',1e-20)=='branch_cut'
    assert limits.view_reason('1','0',1e-40) is None
    for eps in [5,1e-5,1e-12,5e-13,1e-13]:
        assert limits.max_iter_for(eps)==500
    assert limits.ref_digits_for(1e-40)>=60
    assert hp.depth_digits(1e-40)==40 and hp.depth_digits(5)==0
    from deep.perturb import deep_map
    gpu_before = {name for name in sys.modules if name == 'cupy' or name == 'mytet_cuda.gpu'}
    out, info = deep_map('1','0',1e-41)
    assert out.shape==(1920,1080) and not out.any()
    assert info['out_of_range'] and info['reason']=='beyond_L'
    assert {name for name in sys.modules if name == 'cupy' or name == 'mytet_cuda.gpu'} == gpu_before


def test_new_tools_parse_without_gpu_import():
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'deep').glob('*.py')) + list((root/'scripts').glob('perturb_*.py'))
    paths += [root/'tests'/name for name in ('test_perturb_gpu.py','fault_injection_perturb.py','perturb_cases.py')]
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WORKER' for t in node.targets):
                ast.parse(ast.literal_eval(node.value),filename=f'{path}:WORKER')


def test_actual_e1_margin_preserves_threshold_boundary():
    from deep import cpu_ref
    from scripts.perturb_scripts import DEFAULT_POINTS
    p = DEFAULT_POINTS[0]
    x, y, eps, n = p['x0'], p['y0'], 1e-30, 500
    pixels = [(0,0),(0,1079),(480,270),(960,540),(1440,810),(1919,0),(1919,1079)]
    expected = [truth.verdict(str(hp.grid_x(x,eps,a)), str(hp.grid_y(y,eps,b)), n, 80)[0] for a,b in pixels]
    got, ref = cpu_ref.samples(x,y,eps,pixels,n,70)
    control, _ = cpu_ref.samples(x,y,eps,pixels,n,70,ref.pixel,'hypot')
    assert set(expected) == {False, True}, 'real e1 samples must straddle the boundary'
    assert reference.range_issue(ref,n) is None
    assert min(ref.log_abs) < math.log(np.finfo(float).tiny)
    assert [v[0] for v in got] == expected, 'margin must match direct high precision truth'
    assert all(v[1] == 0 for v in got), 'normal tiny returns must not set flag 2'
    assert len({v[0] for v in control}) == 1, 'hypot control collapses onto rounded reference verdict'
    assert all(v[1] == 0 and v[0] == truth.verdict(*ref.center,n,80)[0] for v in control)
    assert [v[0] for v in control] != expected


def test_margin_array_is_rounded_after_high_precision_subtraction():
    ref = reference.orbit('2','0',4,70)
    with mp.workdps(70):
        z, lc = mp.mpc(2), mp.log(2)
        for m in ref.margin[2:]:
            assert m == float(z.real*z.real+z.imag*z.imag-mp.mpf('1e20'))
            z = mp.exp(z*lc)


def test_cpu_tiny_reference_and_pixel_range_rules():
    from deep import cpu_ref
    # A log-preserved tiny reference followed by 1 is normal even when expm1 overflows.
    ref = reference.orbit('1','0',2,60)
    ref.z[3], ref.log_abs[3] = 0, -2000.
    ref.lc = 1000+0j
    assert cpu_ref.pixel(ref,1+0j,1) == (False,0,1)
    ref.log_abs[3] = -1000.
    # A tiny reference with a normal-range pixel value is no longer a range exit — the pixel's
    # value comes from ln|Z| + Re w and arg Z + Im w (log-polar path); it stays a finite, non-escaping z here.
    assert cpu_ref.pixel(ref,1+0j,1) == (False,0,1)
    ref.log_abs[3] = float('nan')
    assert cpu_ref.pixel(ref,1+0j,1) == (False,3,0)
    # A normal next reference and underflowed increment: Lambda below 2^-900 loses information.
    ref = reference.orbit('1','0',1,60)
    ref.lc = -1+0j
    assert cpu_ref.pixel(ref,1e-300+0j,1) == (False,2,0)
    assert cpu_ref.pixel(ref,0j,1) == (False,0,1)
    assert cpu_ref.pixel(ref,1e-40+0j,1) == (False,0,1)


def test_notice_binary_orientation_and_strings():
    from viewer.notice import notice_map, text_mask, range_text, LIMIT_TEXT, BAND_HEIGHT
    assert LIMIT_TEXT == '더 깊이는 지원하지 않음 (하한 L = 1e-40)'
    assert range_text('forced') == '계산 범위 이탈: forced — 이 화면은 계산하지 않음'
    mask = notice_map(None, LIMIT_TEXT)
    assert mask.shape == (1920,1080) and mask.dtype == np.bool_
    # canonical 0/1 bytes, or the renderer's view(uint8)*255 draws the letters black
    assert set(np.unique(mask.view(np.uint8)).tolist()) <= {0, 1}
    assert set(np.unique(notice_map(None, range_text('forced')).view(np.uint8)).tolist()) == {0, 1}
    assert np.count_nonzero(mask) > 0
    assert not mask[:, :1080-BAND_HEIGHT].any()
    base = np.ones((1920,1080), dtype=bool)
    shown = notice_map(base,LIMIT_TEXT)
    assert shown[:, :1080-BAND_HEIGHT].all() and base.all()
    assert np.array_equal(shown[:,1080-BAND_HEIGHT:],text_mask(LIMIT_TEXT)[:,1080-BAND_HEIGHT:])
    from PIL import Image, ImageDraw, ImageFont
    from viewer.notice import FONT
    image = Image.new('1',(1920,1080),0)
    ImageDraw.Draw(image).text((24,22),LIMIT_TEXT,font=ImageFont.truetype(FONT,32),fill=1)
    assert np.array_equal(mask.T[::-1], np.asarray(image,dtype=bool))


def test_viewer_hp_start_and_real_limit_clamp():
    from viewer.deep_state import initial_states, apply_event
    from scripts.perturb_scripts import DEFAULT_POINTS
    p = DEFAULT_POINTS[0]
    state, high = initial_states([0.,0.,1.1e-40],[p['x0'],p['y0']])
    assert state[:2] == (float(p['x0']),float(p['y0']))
    assert high[:2] == (Decimal(p['x0']),Decimal(p['y0']))
    rejected, same_hp, clamped = apply_event(state,high,dict(type='wheel',steps=1))
    assert clamped and rejected is state and same_hp is high
    assert params.apply_wheel(*state,1)[2] < limits.L
    state, high, clamped = apply_event(state,high,dict(type='wheel',steps=-1))
    assert not clamped and state[2].hex() == high[2].hex()
    # No start_hp means exact float centres rather than their short decimal repr.
    state, high = initial_states([.1,.2,5.])
    assert high[:2] == (Decimal(.1),Decimal(.2))


def test_generated_scripts_replay_eps_bits_and_video_corrections():
    from scripts.perturb_scripts import generate, DEFAULT_POINTS, replay
    from viewer.deep_state import initial_states, apply_event
    files, diag = generate(DEFAULT_POINTS[0])
    assert set(files) == {'perturb_cross','perturb_limit','perturb_drag','perturb_video',
                          'perturb_depth_15','perturb_depth_20','perturb_depth_30','perturb_depth_40'}
    for name, data in files.items():
        state, high = initial_states(data['start'],data.get('start_hp'))
        assert [e['t'] for e in data['events']] == sorted(e['t'] for e in data['events'])
        for i, ev in enumerate(data['events']):
            assert ev['i'] == i
            state, high, _ = apply_event(state,high,ev)
            assert high[2].hex() == state[2].hex() and state[2] >= limits.L
    assert replay(files['perturb_cross'])[0][2] < 1e-14
    assert len(replay(files['perturb_limit'])[2]) == 3
    assert diag['end_center'] == [str(v) for v in replay(files['perturb_video'])[1][:2]]
    assert limits.L <= diag['end_state'][2] < 1.02*limits.L
    assert [v['threshold'] for v in diag['corrections']] == [1e-14,1e-29]
    assert max(abs(Decimal(v)) for v in diag['delta_pixels']) < Decimal('.01')


def test_truth_fixture_covers_condition1_matrix():
    """The GPU accuracy fixture is the full 3 positions x 4 depths matrix, no gaps or repeats."""
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).resolve().parent / 'data' / 'perturb_truth_fixture.json').read_text(encoding='utf-8'))
    rows = data['positions']
    keys = [(r['id'], r['eps']) for r in rows]
    want = [(p, e) for p in data['matrix']['positions'] for e in data['matrix']['depths']]
    assert len(data['matrix']['positions']) >= 3 and data['matrix']['depths'] == [1e-15, 1e-20, 1e-30, 1e-40]
    assert sorted(keys) == sorted(want) and len(set(keys)) == len(keys) == 12, keys
    for r in rows:
        assert len(r['diverged']) == 625 and r['max_iter'] == 500 and r['digits'] >= hp.depth_digits(r['eps']) + 20
        share = sum(r['diverged']) / 625
        assert 0.05 <= share <= 0.95, (r['id'], r['eps'], share)


def test_r1_counterexample_restart_value_escapes():
    """C = 1e11 i, eps 1e-20, N = 5 — z_1 = exp(-pi 1e11 / 2) is tiny, then 1, then c with
    |c| > 1e10, so the third power escapes. The perturbation (reference = this pixel, gamma = 0) must agree with
    the direct truth; the removed «reset = non-divergent» shortcut said (False, 0, 5)."""
    from deep import cpu_ref
    assert truth.verdict('0', '100000000000', 5, 40) == (True, 3)
    ref = reference.orbit('0', '100000000000', 5, 40, pixel=(960, 540))
    assert cpu_ref.pixel(ref, 0j, 5, 'margin') == (True, 0, 3)
