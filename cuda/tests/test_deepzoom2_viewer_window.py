"""Opt-in window test: MYTET_DEEPZOOM2_WINDOW=1. Every window uses protected launch.run.

No window or GPU operation happens during collection. Offline final checks use
a separate audited spawn process and the synchronous deep_map API.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

F=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(os.environ.get('MYTET_DEEPZOOM2_WINDOW')!='1',
                             reason='window test: set MYTET_DEEPZOOM2_WINDOW=1 to run')


class OfflineEngine:
    def __init__(self,engine):
        self.engine=engine

    def plan(self,x,y,eps,*,max_iter):
        return x,y,eps,max_iter

    def launch(self,plan,*,stride):
        assert stride==1
        x,y,eps,n=plan
        result=self.engine.deep_map(x,y,eps,max_iter=n)
        return SimpleNamespace(done=lambda:True,result=lambda:result)


def offline_engine_factory(directory,responses):
    from viewer.deep_worker import gpu_engine
    return OfflineEngine(gpu_engine(directory,responses))


def final_render_sha(bits):
    """Inverse of the notice readback rule: map = (rt[...,0] > 127).T."""
    m=np.unpackbits(np.frombuffer(bits,np.uint8),count=1920*1080).reshape(1920,1080)
    rt=np.empty((1080,1920,4),np.uint8)
    rt[...,:3]=(m.T*255)[...,None]
    rt[...,3]=255
    return hashlib.sha256(rt.tobytes()).hexdigest()


def assert_final_targets(frames,expected):
    from viewer.deep_view import state_key
    shown=[f for f in frames if f.get('intermediate') is False
           and f.get('underlying_kind',f.get('present_kind'))=='deep_final']
    assert shown
    for f in shown:
        assert f['source_stride']==1
        assert state_key((f['deep']['x0'],f['deep']['y0'],f['display_params'][2])) == state_key(
            (*f['source_hp'][:2],float.fromhex(f['source_hp'][3])))
        assert f['rt_sha256']==expected[f['source_request_id']],f['seq']


def assert_crossing_result(result):
    assert result['map_comparison_computed']
    assert result['frame_count']>0 and result['excess_axis_count']==0
    assert len(result['frames'])==result['frame_count']
    for f in result['frames']:
        assert all(abs(d)<=max(1,b) for d,b in zip(f['center_diff_px'],f['f64_round_limit_px'])),f
    counted=[r for r in result['crossings'] if r['counted']]
    assert len(counted)==result['counted_crossings']>=6
    for r in counted:
        assert all(abs(v)<=1 for v in r['max_offset']),r
        assert len(r['ties'])==1 or [0,0] in r['ties'],r


@pytest.fixture(scope='module')
def out():
    directory=F/'logs/deepzoom2/window'/f'{dt.datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}'
    directory.mkdir(parents=True,exist_ok=False)
    return directory


def protected_run(script,directory,*,readback='none',env=None):
    from viewer import launch
    from scripts import cpu_guard
    return cpu_guard.guarded_run(lambda d:launch.run(script,'threaded',d,readback=readback,
        max_wait=float(os.environ.get('MYTET_WINDOW_MAX_WAIT','3600')),timeout=600,env=env),directory,attempts=3)


def test_threaded_responsiveness_and_all_final_bits_equal_offline_deep_map(out):
    from scripts.deepzoom2_viewer_measure import run,jsonl
    from viewer.deep_worker import ProcessWorker
    directory=out/'responsive'
    r=run(directory,F/'viewer/scripts/deepzoom2_responsive.json',
          max_wait=float(os.environ.get('MYTET_WINDOW_MAX_WAIT','3600')),timeout=600,readback='precise')
    assert r is not None and r['valid'] and r['final_after_last_input'] and not r['missing_inputs']
    assert r['input_latency_ms']['p95']<=50
    assert r['present_interval_ms']['p99']<=50 and r['present_interval_ms']['max']<=200
    assert r['duplicate_jobs']==r['duplicate_rejections']==r['recompute_after_eviction']==0 and r['final_picture_count']>0
    finals=jsonl(directory/'deep_final_maps.jsonl')
    frames=jsonl(directory/'frames.jsonl')
    assert any(f['underlying_kind']=='deep_warped' for f in frames)
    assert any(f['underlying_kind']=='deep_coarse' for f in frames)
    worker=ProcessWorker(directory/'offline_gpu',offline_engine_factory)
    expected={}
    try:
        with np.load(directory/'deep_final_maps.npz') as maps:
            assert len(maps.files)==len(finals)>0
            for row in finals:
                request={k:row[k] for k in ('id','x0','y0','eps','eps_hex','stride','max_iter','last_input')}
                worker.submit(request)
                end=None
                deadline=time.perf_counter()+180
                while end is None and time.perf_counter()<deadline:
                    for event in worker.poll():
                        assert event['event']!='error',event
                        if event['event']=='end': end=event
                    time.sleep(.005)
                assert end is not None,'offline deep_map timeout'
                assert not end['info']['out_of_range']
                assert np.array_equal(maps[str(row['id'])],np.frombuffer(end['bits'],np.uint8))
                expected[row['id']]=final_render_sha(end['bits'])
        assert_final_targets(frames,expected)
    finally:
        worker.close()


def test_threaded_crossing_centers(out):
    from scripts.deepzoom2_viewer_measure import jsonl
    directory=protected_run(F/'viewer/scripts/deepzoom2_crossing.json',out/'crossing')
    # CLI parent uses perturb_tools.guarded; its GPU child immediately audits its
    # own context, alongside the existing parent launch audit and CPU guard.
    subprocess.run([sys.executable,str(F/'scripts/deepzoom2_crossing.py'),
                    '--directory',str(directory)],cwd=F,check=True,timeout=600)
    r=json.loads((directory/'crossing.json').read_text(encoding='utf-8'))
    assert_crossing_result(r)
    assert r['resync_count']>0
    assert sum(r['crossed_deep'] for r in jsonl(directory/'inputs.jsonl'))>=6


@pytest.mark.parametrize('kind',['limit','range'])
def test_notices_same_text_and_rendered_pixels(out,kind):
    from viewer import notice
    from viewer.deepzoom2_scripts import position
    from scripts.deepzoom2_viewer_measure import jsonl
    point=position()
    script=out/(kind+'.json')
    data=dict(start=[float(point['x0']),float(point['y0']),1.05e-40],
              start_hp=[point['x0'],point['y0']],duration=2.,
              events=[dict(t=1.,type='wheel',steps=1,i=0)] if kind=='limit' else [])
    script.write_text(json.dumps(data),encoding='utf-8')
    directory=protected_run(script,out/kind,readback='all',
                            env={'MYTET_FORCE_RANGE_EXIT':'1'} if kind=='range' else None)
    frames=jsonl(directory/'frames.jsonl')
    shown=[f for f in frames if f.get('notice')==('limit' if kind=='limit' else 'range:forced')]
    assert shown
    expected=notice.LIMIT_TEXT if kind=='limit' else notice.range_text('forced')
    with np.load(directory/'notice_targets.npz') as targets:
        for frame in shown:
            assert frame['notice_text']==expected
            assert frame['notice_rendered_text_pixels']==frame['notice_text_pixels']>0
            actual=np.unpackbits(targets[str(frame['seq'])],count=1920*1080).reshape(1920,1080).astype(bool)
            assert np.all(actual[notice.text_mask(expected)])
