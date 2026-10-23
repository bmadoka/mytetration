"""Pure viewer/tool tests: no CuPy, native window, nvidia-smi or encoders."""
import json
from pathlib import Path
import sys
import uuid
import numpy as np

from scripts.perturb_scripts import F
sys.path.insert(0, str(F/'scripts'))
from scripts.perturb_measure import summarize
from viewer.make_perturb_video import compose, path_label, choose_crf


def test_measure_reports_new_compute_and_present_intervals():
    directory = F/'logs/perturb/cpu_test_cache'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    frames = [{'compute_path':'perturb','compute_ms':ms,'t_present':t,'kind':kind,'last_input':i,
               'display_params':[0.,0.,e],'display_params_hex':[(0.).hex(),(0.).hex(),e.hex()],
               'deep': {'out_of_range':False,'reason':None}}
              for ms,t,kind,i,e in [(100,0,'initial',-1,1e-20),(40,1,'fast',0,1.2e-20),(60,1.1,'fast',5,2e-20),
                                    (80,1.3,'fast',9,1.1e-20),(120,2,'precise',9,1e-20)]]
    (directory/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
    (directory/'contamination.json').write_text(json.dumps({'verdict':{'contaminated':False}}),encoding='utf-8')
    r = summarize(directory)
    assert r['deep_frames'] == 5 and r['compute_ms'] == {'median':80.,'p95':116.,'max':120}
    assert np.allclose(r['moving_present_intervals_s'],[.1,.2])
    assert np.isclose(r['moving_fps'],1/.15)
    assert r['contamination']['contaminated'] is False
    # exact start depth reported separately from the mixed-eps statistics
    assert r['start_eps'] == 1e-20 and r['start_depth_frames'] == 2 and r['start_depth_compute_ms']['median'] == 110.
    assert r['eps_range'] == [1e-20, 2e-20]


def test_video_overlay_path_labels_and_crf_cap():
    from PIL import ImageFont
    from viewer.notice import FONT
    labels = [('f32','빠른 f32'),('f64','빠른 f64'),('double','정밀 double'),('perturb','섭동')]
    fonts = (ImageFont.truetype(FONT,17),ImageFont.truetype(FONT,18))
    bits = np.packbits(np.zeros((540,960),dtype=bool))
    for path, label in labels:
        frame = {'compute_path':path,'kind':'fast','params':[0.,0.,1e-30]}
        assert path_label(frame) == label
        out = compose(bits,frame,1.,[1.],[],1.,fonts)
        assert out.shape == (540,960,3) and out.dtype == np.uint8
        assert np.count_nonzero(out[:146]) > 0 and not out[146:].any()
    selected, attempts = choose_crf(lambda crf: 48_000_001 if crf < 30 else 48_000_000,48_000_000)
    assert selected == 30 and [a['crf'] for a in attempts] == [28,29,30]


def test_changed_viewer_and_tools_parse_without_gpu():
    import ast
    for rel in ('viewer/app.py','viewer/launch.py','viewer/notice.py','viewer/deep_state.py',
                'viewer/make_perturb_video.py','tests/test_viewer.py','tests/test_perturb_gpu.py'):
        ast.parse((F/rel).read_text(encoding='utf-8'),filename=rel)
    # checked in a fresh interpreter (the full suite imports CuPy in earlier GPU tests)
    import subprocess
    code = ("import sys; sys.path[:0]=[r'%s', r'%s']; import viewer.notice, viewer.deep_state, deep.hp, deep.limits, "
            "deep.reference, deep.truth, deep.cfn; print(int('cupy' in sys.modules or 'mytet_cuda.gpu' in sys.modules))"
            % (F, F/'viewer'))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == '0', out


def test_deep_frame_requests_async_stages_and_archives_with_fake_compute(monkeypatch):
    """New orchestration still verifies distinct HP centres, raw final bits and notice pixels."""
    from test_deepzoom2_viewer_cpu import fake_viewer
    from viewer import notice
    from deep import hp
    monkeypatch.delenv('MYTET_FORCE_RANGE_EXIT',raising=False)
    v=fake_viewer()
    v.readback='all'
    v.frame('initial')
    worker=v.deep_controller.worker
    first=worker.submitted[-1]
    worker.complete()
    v.frame('precise')
    assert worker.submitted[-1]['stride']==1
    worker.complete()
    v.frame('precise')
    final=v.deep_controller.final_for(v.hp)
    assert final['map'].shape==(1920,1080)
    assert np.array_equal(np.packbits(final['map']),np.frombuffer(final['bits'],np.uint8))
    import hashlib
    assert v.frames[-1]['map_sha256']==hashlib.sha256(final['map'].tobytes()).hexdigest()
    assert not v.frames[-1]['intermediate']
    v.on_input(dict(type='drag',dx=-100,dy=0))
    monkeypatch.setenv('MYTET_FORCE_RANGE_EXIT','1')
    v.frame('precise')
    assert worker.submitted[-1]['x0']!=first['x0']
    assert all(f['compute_path']=='perturb' for f in v.frames)
    assert v.frames[-1]['notice']=='range:forced'
    expected=notice.notice_map(None,notice.range_text('forced'))
    assert np.array_equal(v.disp.image,expected)
    assert v.frames[-1]['notice_rendered_text_pixels']==v.frames[-1]['notice_text_pixels']>0
    units=[v.deep_controller._unit(r) for r in worker.submitted]
    assert len(units)==len(set(units))
