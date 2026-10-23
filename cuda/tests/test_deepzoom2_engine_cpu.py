"""CPU numerical contracts for the extended-orbit rebasing engine."""
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid
import mpmath as mp
import numpy as np
import pytest
from deep import cpu_ref,hp,limits,perturb,reference,truth

F = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('eps,n',[(1e-13,500),(9.99e-14,500),(1e-15,540),(1e-20,640),  # 9.99e-14: smooth rule (was 520)
                                  (1e-25,740),(1e-30,840),(1e-35,940),(1e-40,1040)])
def test_depth_budget(eps,n):
    assert limits.max_iter_for(eps)==n
    assert limits.ref_digits_for(eps)==hp.depth_digits(eps)+30
    assert limits.L==1e-40


def test_plan_import_is_cpu_only():
    code = "from deep import perturb; p=perturb.plan('1','.2',1e-40,max_iter=2); import sys; assert 'cupy' not in sys.modules and 'mytet_cuda.gpu' not in sys.modules; assert p.ref is not None"
    result = subprocess.run([sys.executable,'-c',code],cwd=F,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    p = perturb.plan('0','0',1e-20)
    job = perturb.launch(p,stride=3)
    assert job.done()
    m,info = job.result(return_stop=True)
    assert m.shape==(640,360) and info['reason']=='near_zero'
    assert info['stop'].shape==m.shape
    with pytest.raises(ValueError):
        perturb.launch(p,stride=1.5)


def test_extended_bootstrap_and_no_initial_escape():
    ref = reference.orbit('0','100000000000',5,70)
    e,delta,escaped,flag,rebased = cpu_ref.step(ref,0,0j,0j,ref.lc,-1)
    assert (e,delta,escaped,flag,rebased)==(1,0j,False,0,False)
    e,delta,escaped,flag,rebased = cpu_ref.step(ref,e,delta,0j,ref.lc,0)
    assert e==2 and ref.z[e]+delta==complex(0,1e11)
    assert not escaped and flag==0  # |c|>R at n=0 is deliberately not tested
    assert cpu_ref.pixel(ref,0j,5)==(True,0,3)
    assert truth.verdict('0','100000000000',5,70)==(True,3)
    ref = reference.orbit('2','0',5,70)
    assert list(ref.z[:4])==[0j,1+0j,2+0j,4+0j]
    assert cpu_ref.pixel(ref,0j,5)==(True,0,4)


def test_cancellation_rebases_by_product_and_exact_cycle():
    ref = reference.orbit('2','0',40,70)
    result,stats = cpu_ref.pixel(ref,-1+0j,40,diagnostics=True)
    assert result==(False,0,40)==cpu_ref.pixel(ref,-1+0j,40,cycles=False)
    assert result[::2]==truth.verdict('1','0',40,70)
    assert stats['rebases']>=1 and stats['cycle_exits']==1
    reset = [r for r in stats['trace'] if r['rebased']]
    assert reset[0]['e']==0 and reset[0]['delta']==1+0j
    assert stats['trace'][-1]['e']==2
    assert cpu_ref.same(stats['trace'][1]['delta'],stats['trace'][-1]['delta'])
    assert not cpu_ref.same(complex(0.,0.),complex(-0.,0.))
    assert not cpu_ref.same(1+0j,1.0000000000000002+0j)


def test_escaped_reference_log_polar_rebases_instead_of_flag1():
    ref = reference.orbit('2','0',40,70)
    assert ref.n_ref==4 and math.isinf(ref.z[-1].real)
    e,delta,escaped,flag,rebased = cpu_ref.step(ref,len(ref.z)-2,-65536+0j,0j,ref.lc,4)
    assert e==0 and rebased and flag==0 and not escaped
    assert abs(delta-1)<1e-10


def test_tiny_cancellation_has_no_exemption_and_uncertainty_is_flag3():
    ref = reference.orbit('1','0',4,70)
    ref.z[3] = 1e-200+0j
    ref.log_abs[3] = math.log(1e-200)
    ref.margin[3] = -1e20
    # expm1(-2) is a cancellation even though the reference is extremely small.
    e,delta,escaped,flag,rebased = cpu_ref.step(ref,2,-2+0j,0j,1+0j,1)
    assert e==0 and rebased and flag==0 and not escaped
    assert math.isclose(delta.real,1e-200*math.exp(-2),rel_tol=1e-14)
    ref.z[3] = 0j
    ref.log_abs[3],ref.log_abs_lo[3] = math.log(1e10),0.
    assert cpu_ref.step(ref,2,0j,0j,0j,1)[3]==3
    ref.log_abs[3] = math.nan
    assert cpu_ref.step(ref,2,0j,0j,0j,1)[3]==3
    for path in (F/'deep').iterdir():
        if path.suffix in ('.py','.cu'):
            text = path.read_text(encoding='utf-8')
            assert 'glitch_abs' not in text,path
            assert 'reset_log' not in text,path
            assert 'AUX_ROUNDS' not in text,path


@pytest.mark.parametrize('x,y',[('1','0'),('0','0'),('0','100000000000'),('2','0'),
                               ('.2','.3'),('-.3','.8'),('-1.1','.18')])
def test_truth_shortcut_equals_plain(x,y):
    assert truth.verdict(x,y,160,70)==truth._verdict(x,y,160,70,False)


@pytest.mark.parametrize('x,y',[('1','0'),('0','100000000000'),('-1.0851625774713976413892441561578844795','.18217150566722271848774639489185212')])
def test_exact_reference_copy_matches_all_direct_bits(x,y):
    a = reference.orbit(x,y,500,50,copy_cycles=True)
    b = reference.orbit(x,y,500,50,copy_cycles=False)
    for name in ('z','log_abs','log_abs_lo','arg','arg_lo','margin'):
        u,v = getattr(a,name),getattr(b,name)
        assert np.array_equal(u.view(np.uint64),v.view(np.uint64)),name
    assert a.n_ref==b.n_ref
    if x=='1':
        assert a.cycle is not None


def test_double_double_input_preserves_large_log_low_bits():
    ref = reference.orbit('0','100000000000',5,70)
    ctx = mp.mp.clone(); ctx.dps=70
    want = -ctx.pi*ctx.mpf('1e11')/2
    hi,lo = ref.log_abs[3],ref.log_abs_lo[3]
    assert abs(ctx.mpf(hi)-want)>ctx.mpf('1e-7')
    assert abs(ctx.mpf(hi)+ctx.mpf(lo)-want)<ctx.mpf('1e-20')
    h,l = cpu_ref.ddadd(hi,lo,-hi)
    assert h+l==lo


def test_accuracy_reference_period_copy_on_actual_m37_orbit():
    data = json.loads((F/'tests/data/deepzoom2_positions.json').read_text())
    pos = data['positions'][0]
    a = reference.at_pixel(pos['x0'],pos['y0'],1e-40,(0,0),1040,70)
    b = reference.orbit(*a.center,1040,70,copy_cycles=False)
    assert a.cycle is not None and a.cycle[1]>1
    for name in ('z','log_abs','log_abs_lo','arg','arg_lo','margin'):
        assert np.array_equal(getattr(a,name).view(np.uint64),getattr(b,name).view(np.uint64)),name


def test_ladder_all_branches_stop_disagreement_and_cache(monkeypatch):
    points = [(str(i),'0') for i in range(4)]
    def fake(x,y,n,d):
        i = int(x)
        if i==0: return True,5
        if i==1: return (False,n) if d==35 else (True,7)
        if i==2: return (True,5) if d==35 else ((True,6) if d==65 else (False,n))
        return False,n
    monkeypatch.setattr(truth,'verdict',fake)
    directory = F/'logs/deepzoom2/cpu_test_cache'/uuid.uuid4().hex
    result = truth.ladder(points,20,15,1,directory)
    a,b,c,d = result['points']
    assert a['decided'] and a['digits']==65 and a['stop']==5
    assert b['decided'] and b['digits']==125 and b['stop']==7
    assert c['unstable'] and not c['decided'] and c['stop'] is None
    assert d['decided'] and d['stop']==20
    assert result['computed_verdicts']==10 and result['d3_points']==2
    monkeypatch.setattr(truth,'verdict',lambda *a:pytest.fail('cache was not used'))
    again = truth.ladder(points,20,15,1,directory)
    assert again['points']==result['points'] and again['cached_verdicts']==10


def test_ladder_spawn_and_grid_coordinate_strings():
    pixels = [(0,0),(960,540),(1919,1079)]
    points = truth.pixel_points('1','.2',1e-40,pixels)
    assert points[0][0]==str(hp.grid_x('1',1e-40,0))
    a = truth.ladder(points,5,40,1)
    b = truth.ladder(points,5,40,2)
    assert a['points']==b['points'] and all(v['decided'] for v in a['points'])
    assert truth.verdict('2','0',3,50)==(False,3)
    assert truth.verdict('2','0',4,50)==(True,4)


def test_populations_determinism_band_definition_and_tile_ownership():
    seed = truth.seed_for('m37',1e-40)
    assert seed==truth.seed_for('m37',1e-40)!=truth.seed_for('m191',1e-40)
    uniform = truth.uniform_pixels(seed,2000)
    assert uniform==truth.uniform_pixels(seed,2000) and len(set(uniform))==2000
    assert len(truth.SAMPLE)==len(set(truth.SAMPLE))==625
    m = np.zeros((1920,1080),dtype=bool)
    m[20:,:] = True
    mask = truth.band_mask(m)
    assert np.all(mask[16:23,:]) and mask.sum()==7*1080
    band = truth.band_pixels(m,seed,5000)
    assert band==truth.band_pixels(m,seed,5000) and len(set(band))==5000
    assert all(mask[p,q] for p,q in band)
    # Both axes, borders, and union checked against a literal independent set.
    small = np.zeros((11,9),dtype=bool); small[1:5,0:4]=True
    expected = set()
    for x in range(11):
        for y in range(9):
            for dx,dy in ((1,0),(0,1)):
                if x+dx<11 and y+dy<9 and small[x,y]!=small[x+dx,y+dy]:
                    expected.update((x+dx*k,y+dy*k) for k in range(-3,4)
                                    if 0<=x+dx*k<11 and 0<=y+dy*k<9)
    assert set(map(tuple,np.argwhere(truth.band_mask(small))))==expected
    from scripts.deepzoom2_eval import structure
    metric = structure(m)
    assert metric['boundary_pairs']==1080 and metric['boundary_tiles']==54
    assert metric['tile_ratio']==54/5184
    assert truth.band_pixels(np.zeros((1920,1080),bool),seed,500)==[]


@pytest.mark.parametrize('eps',[1e-20,1e-40])
def test_cpu_accuracy_m37_against_direct_truth(eps):
    positions = json.loads((F/'tests/data/deepzoom2_positions.json').read_text())
    pos = next(p for p in positions['positions'] if p['id']=='m37')
    n = limits.max_iter_for(eps)
    ref,_,_ = reference.choose(pos['x0'],pos['y0'],eps,n,limits.ref_digits_for(eps))
    gx,gy = reference.offsets(pos['x0'],pos['y0'],eps,ref)
    pixels = truth.uniform_pixels(123,24)
    targets = truth.ladder(truth.pixel_points(pos['x0'],pos['y0'],eps,pixels),n,hp.depth_digits(eps),1)['points']
    for (x,y),target in zip(pixels,targets):
        got,stats = cpu_ref.pixel(ref,complex(gx[x],gy[y]),n,diagnostics=True)
        assert target['decided'] and not target['unstable']
        assert got==(target['diverged'],0,target['stop']),((x,y),got,target)
        assert cpu_ref.pixel(ref,complex(gx[x],gy[y]),n,cycles=False)==got


def test_accuracy_cpu_n6_known_counterexample_current_budget():
    x = '0.37610494333279680704622167329529150248935039028424858371281725'
    y,eps = '1.9671747211895914',1e-15
    n = limits.max_iter_for(eps)
    point = truth.pixel_points(x,y,eps,[(953,543)])
    target = truth.ladder(point,n,15,1)['points'][0]
    assert target['decided'] and not target['unstable']
    values,ref = cpu_ref.samples(x,y,eps,[(953,543)],n,limits.ref_digits_for(eps))
    assert values[0]==(target['diverged'],0,target['stop'])


def _fault_records():
    baseline = [dict(nodeid='tests/test_deepzoom2_engine_cpu.py::test_accuracy', when=phase,
                     outcome='passed', xfail=False) for phase in ('setup', 'call', 'teardown')]
    faulty = [dict(row) for row in baseline]
    faulty[1].update(outcome='failed', exc_type='builtins.AssertionError', raised_in_test_body=True,
                     at_assert_statement=True, message_first_line='truth differs')
    return baseline, faulty


def test_tools_fault_preflight_uses_exact_replacements_and_restores_copies(tmp_path):
    import fault_injection_deepzoom2 as fi
    originals = {rel: (F/rel).read_bytes() for rel, _ in fi.FAULTS.values()}
    result = fi.preflight(tmp_path)
    assert [row['replacement'] for row in result['faults']] == [
        'period = max(1, e-first-1)', 'return true;', 'return false;', 'return false;']
    assert all(r['sha_restored'] and r['exceptional_restore_exercised'] for r in result['faults'])
    assert all((F/rel).read_bytes() == raw for rel, raw in originals.items())
    root = next(tmp_path.glob('source_copy_*'))
    for number in fi.FAULTS:
        path, _, old, new = fi.anchor(root, number)
        assert old != new and old[:len(old)-len(old.lstrip())] == new[:len(new)-len(new.lstrip())]
        if path.suffix == '.py':
            with fi.mutation(root, number, {}):
                compile(path.read_bytes(), str(path), 'exec')
    with (root/'deep'/'duplicate.py').open('w') as stream:
        stream.write('# FAULT_1\n')
    with pytest.raises(ValueError, match='unique'):
        fi.anchor(root, 1)


@pytest.mark.parametrize('change', ['setup', 'helper', 'raise', 'exception', 'collection', 'skip', 'missing', 'exit', 'xfail'])
def test_tools_fault_judgement_rejects_nonassertion_or_incomplete_runs(change):
    import fault_injection_deepzoom2 as fi
    baseline, faulty = _fault_records()
    assert fi.baseline_ok(0, baseline) and fi.judge(1, faulty, baseline)['caught']
    code = 1
    if change == 'setup': faulty[1]['when'] = 'setup'
    if change == 'helper': faulty[1]['raised_in_test_body'] = False
    if change == 'raise': faulty[1]['at_assert_statement'] = False
    if change == 'exception': faulty[1]['exc_type'] = 'builtins.ValueError'
    if change == 'collection': faulty.append(dict(nodeid='bad.py', when='collect', outcome='failed', xfail=False))
    if change == 'skip': faulty[2]['outcome'] = 'skipped'
    if change == 'missing': baseline += [dict(baseline[1], nodeid='missing')]
    if change == 'exit': code = 2
    if change == 'xfail': faulty[2]['xfail'] = True
    assert not fi.judge(code, faulty, baseline)['caught']


def test_tools_fault_runner_gates_baseline_and_restores_when_pytest_raises(tmp_path, monkeypatch):
    import shutil
    import fault_injection_deepzoom2 as fi
    root = tmp_path/'repo'
    (root/'deep').mkdir(parents=True)
    for rel, _ in fi.FAULTS.values():
        shutil.copyfile(F/rel, root/rel)
    original = {p: p.read_bytes() for p in (root/'deep').iterdir()}
    monkeypatch.setattr(fi, 'F', root)
    baseline, faulty = _fault_records()
    monkeypatch.setattr(fi, 'run_pytest', lambda *args: (1, faulty))
    with pytest.raises(RuntimeError, match='normal run'):
        fi.run(tmp_path/'bad_baseline')
    assert all(p.read_bytes() == b for p, b in original.items())
    calls = []
    def failure(directory, label):
        calls.append(label)
        if label == 'baseline': return 0, baseline
        raise RuntimeError('worker failed')
    monkeypatch.setattr(fi, 'run_pytest', failure)
    with pytest.raises(RuntimeError, match='worker failed'):
        fi.run(tmp_path/'exception')
    report = json.loads((tmp_path/'exception/faults.json').read_text())
    assert report['faults'][0]['sha_restored'] and not report['all_faults_caught']
    assert all(p.read_bytes() == b for p, b in original.items())
    monkeypatch.setattr(fi, 'run_pytest', lambda *args: (0, baseline))
    with pytest.raises(RuntimeError, match='not caught'):
        fi.run(tmp_path/'missed')
    report = json.loads((tmp_path/'missed/faults.json').read_text())
    assert len(report['faults']) == 4 and all(r['sha_restored'] for r in report['faults'])
    assert not report['all_faults_caught']
    monkeypatch.setattr(fi, 'run_pytest', lambda d, label: (0, baseline) if label == 'baseline' else (1, faulty))
    assert fi.run(tmp_path/'caught')['all_faults_caught']
    assert all(p.read_bytes() == b for p, b in original.items())


def test_tools_video_tour_hp_target_grid_wheels_and_all_six_holds():
    from viewer.deepzoom2_scripts import position, VIDEO_POSITIONS, VIDEO_DEPTHS, video_script
    point = position(VIDEO_POSITIONS, 'm37')
    script = video_script(point)
    state = hp.from_floats(*script['start'])
    previous_wheel = -math.inf
    assert script['start'] == [0., 0., 5.]
    assert [h['depth'] for h in script['holds']] == list(VIDEO_DEPTHS)
    for i, ev in enumerate(script['events']):
        assert ev['i'] == i and abs(ev['t']/.016-round(ev['t']/.016)) < 1e-9
        if i: assert ev['t']-script['events'][i-1]['t'] >= .016-1e-9
        if ev['type'] == 'wheel':
            assert ev['t']-previous_wheel >= .1-1e-9
            assert ev['steps']==1. or 0 < ev['steps'] <= 1/6
            previous_wheel = ev['t']
        from viewer.deep_state import apply_event, project
        _,state,clamped=apply_event(project(state),state,ev)
        assert not clamped
        assert state[2] >= limits.L
        if i >= script['zoom_start_event']:
            for k, step in enumerate(hp.spacing(state[2])):
                assert abs(hp.CTX.divide(hp.CTX.subtract(state[k], hp.D(point[('x0','y0')[k]])), step)) <= 3
    for h in script['holds']:
        assert h['end']-h['start'] >= 1.5-1e-9 and h['depth'] <= h['eps'] < 1.02*h['depth']
        assert not any(h['start'] < e['t'] < h['end'] for e in script['events'])
    assert state[2].hex() == limits.L.hex()
    assert hp.to_record(state) == script['end_hp']
    checked=json.loads((Path(__file__).resolve().parents[1]/'viewer/scripts/deepzoom2_video.json').read_text(encoding='utf-8'))
    assert checked['events']==script['events'] and checked['end_hp']==script['end_hp']


def _synthetic_video_recording(tmp_path):
    """Artificial window bits and timing; never represents an actual viewer run."""
    from viewer.deepzoom2_scripts import VIDEO_DEPTHS
    root = tmp_path/'synthetic_recording'
    root.mkdir()
    holds, frames = [], [dict(params=[0., 0., 5.], t_present=100., kind='initial', compute_path='double')]
    for i, eps in enumerate(VIDEO_DEPTHS):
        at = 2.+2*i
        holds.append(dict(depth=eps, eps_hex=eps.hex(), start=at, end=at+1.504, hp=['-1', '.2']))
        frames.append(dict(params=[-1., .2, eps], t_present=100.+at+1, kind='fast',
                           compute_path='perturb', underlying_kind='deep_final', present_kind='deep_hold',
                           intermediate=False, deep=dict(x0='-1', y0='.2')))
    script = root/'script.json'
    script.write_text(json.dumps(dict(holds=holds)), encoding='utf-8')
    meta = dict(mode='threaded', script=str(script), t_playback_start=100., t_end=114.,
                monitor_refresh_hz=2., input_t={}, start_hp=['0','0'], end_hp=['-1','.2'],
                recorded_frames=len(frames), recorded_non_binary_pixels=0)
    (root/'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    (root/'frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames), encoding='utf-8')
    (root/'contamination.json').write_text(json.dumps(dict(verdict=dict(contaminated=False))), encoding='utf-8')
    pixels = np.zeros((len(frames), 540, 960), dtype=np.uint8)
    pixels[:, :100, 400:500] = 1  # bottom in the recording, top-to-bottom reversal is required
    np.save(root/'window_frames_bits.npy', np.packbits(pixels.reshape(len(frames), -1), axis=1))
    return root


def test_tools_video_composition_uses_window_bits_and_deep_hold_labels(tmp_path):
    from PIL import ImageFont
    from viewer import make_deepzoom2_video as mv
    root = _synthetic_video_recording(tmp_path)
    meta, frames, bits, _, stops = mv.load_recording(root)
    fonts = (ImageFont.truetype(mv.FONT, 17), ImageFont.truetype(mv.FONT.replace('malgun','malgunbd'), 18))
    rgb = mv.compose(bits[-1], frames[-1], 113., [100.,113.], [], 100., fonts)
    expected = np.unpackbits(bits[-1]).reshape(540,960)[::-1]*np.uint8(255)
    assert rgb.shape == (540,960,3) and rgb.dtype == np.uint8
    assert np.array_equal(rgb[146:,:,0], expected[146:])
    assert rgb[500,450].tolist() == [255,255,255] and rgb[400,450].tolist() == [0,0,0]
    assert len(stops) == 6 and mv.path_label(frames[-1]) == '깊은 최종'
    for kind, label in [('deep_coarse','깊은 거친'), ('deep_warped','깊은 옮김'), ('deep_notice','깊은 안내')]:
        assert mv.path_label(dict(frames[-1], underlying_kind=kind)) == label
    frames[-1]['intermediate'] = True
    (root/'frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames), encoding='utf-8')
    with pytest.raises(ValueError, match='actual final'):
        mv.load_recording(root)


@pytest.mark.parametrize('eps,accepted',[
    (limits.L,True),(math.nextafter(limits.L,math.inf),False),(limits.L*(1+1e-9),False),
    (math.nextafter(limits.L,0.),False),(limits.L*(1+2e-9),False),(1.0122980471535644e-40,False)])
def test_tools_video_end_depth_exact_bits_and_numeric_state_key(tmp_path,eps,accepted):
    from viewer import make_deepzoom2_video as mv
    root=_synthetic_video_recording(tmp_path)
    frames=[json.loads(s) for s in (root/'frames.jsonl').read_text().splitlines()]
    frames[-1]['params'][2]=eps
    frames[-1]['deep']=dict(x0='-1.0000',y0='2.00E-1')
    (root/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)),encoding='utf-8')
    script=json.loads((root/'script.json').read_text())
    script['holds'][-1]['eps_hex']=eps.hex()
    (root/'script.json').write_text(json.dumps(script),encoding='utf-8')
    if accepted:
        assert len(mv.load_recording(root)[-1])==6
    else:
        with pytest.raises(ValueError,match='tour must finish in'):
            mv.load_recording(root)


def test_tools_video_encoder_crf_cap_probe_and_rejects_missing_stop(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from viewer import make_deepzoom2_video as mv
    root = _synthetic_video_recording(tmp_path)
    monkeypatch.setattr(mv, 'SEND_BYTES', 100)
    calls = []
    class Pipe:
        count = 0
        def write(self, b): self.count += len(b)
        def close(self): pass
    class Encoder:
        def __init__(self, command, **kw):
            self.stdin = Pipe()
            self.path = Path(command[-1])
            self.crf = int(command[command.index('-crf')+1])
            calls.append(self)
        def wait(self):
            self.path.write_bytes(bytes(101 if self.crf == 28 else 100))
            return 0
    def probe(command, **kw):
        return SimpleNamespace(stdout=json.dumps(dict(streams=[dict(codec_type='video',codec_name='h264',
            width=960,height=540,nb_read_frames=28)],format=dict(duration='14.0',size=str(Path(command[-1]).stat().st_size)))))
    monkeypatch.setattr(mv.subprocess, 'Popen', Encoder)
    monkeypatch.setattr(mv.subprocess, 'run', probe)
    result = mv.video(root, tmp_path/'encoded')
    assert [e.crf for e in calls] == [28,29]
    assert all(e.stdin.count == 28*960*540*3 for e in calls)
    assert result['outputs']['archive']['crf'] == 28 and result['outputs']['send']['size_bytes'] == 100
    assert result['video_frames'] == 28 and result['duration_s'] == 14.
    assert len(result['depth_stops']) == 6
    with pytest.raises(FileExistsError): mv.video(root, tmp_path/'encoded')
    text = (root/'frames.jsonl').read_text().splitlines()
    frame = json.loads(text[2]); frame['underlying_kind'] = 'deep_coarse'; text[2] = json.dumps(frame)
    (root/'frames.jsonl').write_text('\n'.join(text), encoding='utf-8')
    with pytest.raises(ValueError, match='no final picture during depth stop'):
        mv.load_recording(root)


def test_tools_scene_collection_copies_evaluator_png_bytes_and_seven_columns(tmp_path):
    from PIL import Image
    from viewer import make_deepzoom2_video as mv
    source = tmp_path/'evaluation'; source.mkdir()
    rows = []
    Image.new('RGB', (1920,1080), 'white').save(source/'initial.png')
    for name in ('m37','m191'):
        for eps in mv.VIDEO_DEPTHS:
            Image.new('RGB', (1920,1080), 'black').save(source/f'{name}_{eps:g}.png')
            rows.append(dict(id=name,eps=eps))
    (source/'eval.json').write_text(json.dumps(dict(rows=rows)), encoding='utf-8')
    result = mv.collect_scenes(source, tmp_path/'collected')
    assert len(result['scenes']) == 14
    for row in result['scenes']:
        assert Path(row['file']).read_bytes() == Path(row['source']).read_bytes()
    with Image.open(result['contact_sheet']) as im:
        assert im.size == (7*240,2*162)
    with pytest.raises(FileExistsError): mv.collect_scenes(source, tmp_path/'collected')
    (source/'eval.json').write_text(json.dumps(dict(rows=rows[:-1])), encoding='utf-8')
    with pytest.raises(ValueError, match='one scene per position and depth'):
        mv.collect_scenes(source, tmp_path/'missing')


def test_tools_video_final_steps_fail_when_exact_depth_unreachable(monkeypatch):
    from viewer import deepzoom2_scripts as scripts
    monkeypatch.setattr(scripts.hp,'apply_wheel',lambda state,steps:(state[0],state[1],math.nextafter(limits.L,math.inf)))
    with pytest.raises(ValueError,match='exactly L'):
        scripts.exact_final_steps(hp.from_floats(0.,0.,1.01e-40))
