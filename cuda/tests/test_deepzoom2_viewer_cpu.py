"""No native window, CUDA, nvidia-smi or device import; real spawn uses a CPU engine."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deep import hp
from viewer import app, deep_state, deep_view
from viewer.deep_worker import DeepController, ProcessWorker
from viewer.deepzoom2_scripts import generate,position,validate


class FakeTransport:
    """Tick-driven delayed CPU work. No completion until explicitly released."""
    def __init__(self):
        self.submitted=[]
        self.events=[]
        self.active=None

    def submit(self,request):
        assert self.active is None
        self.active=request
        self.submitted.append(request)
        self.events.append(dict(event='start',**request,t_start=time.perf_counter()))

    def complete(self,out_of_range=False):
        r=self.active
        stride=r['stride']
        m=np.zeros(((1920+stride-1)//stride,(1080+stride-1)//stride),bool)
        m[1::2,::2]=True
        self.events.append(dict(event='end',**r,bits=np.packbits(m).tobytes(),shape=m.shape,
            map_sha256=hashlib.sha256(m.tobytes()).hexdigest(),
            info=dict(path='fake',out_of_range=out_of_range,reason='fake_range' if out_of_range else None,
                      max_iter=r['max_iter'],t_ms={'total':50.}),t_start=time.perf_counter()-.05,t_end=time.perf_counter()))
        self.active=None

    def poll(self):
        events,self.events=self.events,[]
        return iter(events)

    def close(self):
        pass


class CpuEngine:
    def __init__(self):
        self.plans=0

    def plan(self,x,y,eps,**kw):
        self.plans+=1
        return self.plans

    def launch(self,plan,*,stride):
        start=time.perf_counter()
        def result():
            return (np.zeros(((1920+stride-1)//stride,(1080+stride-1)//stride),bool),
                    dict(path='fake_cpu',plan_index=plan,out_of_range=False,reason=None))
        return SimpleNamespace(done=lambda:time.perf_counter()-start>=.025,result=result)


def cpu_engine_factory(directory,responses):
    responses.put(dict(event='fake_ready',pid=os.getpid(),cupy_imported='cupy' in sys.modules))
    return CpuEngine()


class FakeDisplay:
    def __init__(self):
        self.image=np.zeros((1920,1080),bool)
        self.presents=0

    def draw(self,m):
        self.image=m.copy()

    def draw_deep(self,m,display,source,stride,source_id):
        self.image,tr=deep_view.warp_map(m,display,source,stride)
        return tr

    def hold_deep(self):
        pass

    def present(self):
        self.presents+=1
        return 0

    def read_render_target(self):
        return np.repeat((self.image.T.astype(np.uint8)*255)[...,None],4,axis=2)

    def read_back_buffer(self):
        return self.read_render_target()[::2,::2,:3]


def fake_viewer(eps=1e-40):
    v=app.Viewer.__new__(app.Viewer)
    point=position()
    v.state,v.hp=deep_state.initial_states([0,0,eps],[point['x0'],point['y0']])
    v.display=v.prev_state=v.state
    v.frames=[]
    v.applied=v.drawn=-1
    v.last_input_t=v.first_new_input_t=v.limit_until=None
    v.input_t={}
    v.input_log=[]
    v.readback,v.mode,v.log_dir='none','threaded',None
    v.deep_controller=DeepController(FakeTransport())
    v.disp=FakeDisplay()
    v.notice_targets={}
    return v


def test_one_hp_center_after_hundreds_of_drags_and_crossings():
    v=fake_viewer(3e-13)
    scripts=generate(position())
    events=scripts['deepzoom2_crossing']['events']
    events += [dict(type='drag',dx=20*(-1 if i%2 else 1),dy=0) for i in range(500)]
    for ev in events:
        before=v.hp
        v.on_input(ev)
        assert v.hp==hp.apply_event(before,ev)
        assert v.state[2].hex()==v.hp[2].hex()
        record=v.input_log[-1]
        if record['resync']:
            assert v.state==deep_state.project(v.hp)
            assert any(abs(hp.D(d))>hp.D('0.5') for d in record['pre_resync_center_diff_px'])
        else:
            assert all(abs(d)<=hp.D('0.5') for d in deep_state.center_diff(v.state,v.hp))
        diag=deep_state.diagnostics(v.state,v.hp)
        assert all(abs(d)<=max(1,b) for d,b in zip(diag['center_diff_px'],diag['f64_round_limit_px']))
        assert v.input_log[-1]['before_hp']==hp.to_record(before)
    assert sum(r['crossed_deep'] for r in v.input_log)>=6
    assert sum(r['resync'] for r in v.input_log)>0
    zero=hp.from_strings('0','0',1e-40)
    assert all(v>0 for v in deep_state.diagnostics(deep_state.project(zero),zero)['f64_round_limit_px'])


def test_deep_indices_at_1e40_drag_zoom_and_coarse_expansion():
    origin=hp.from_strings('-1.08516257747139764138924415615788447953747633110966441982286','.1821715056672227184877463948918521197378646800507803',1e-40)
    moved=hp.apply_drag(origin,20,-20)
    ix,iy,tr=deep_view.warp_indices(moved,origin)
    assert np.array_equal(ix[20:],np.arange(1900))
    assert np.array_equal(iy[20:],np.arange(1060))
    assert np.allclose(tr['offset'],[-20,-20])
    zoom=hp.apply_wheel(origin,1)
    ix,iy,tr=deep_view.warp_indices(zoom,origin)
    assert tr['scale']==pytest.approx(.9)
    assert ix[0]==96 and ix[-1]==1823
    coarse=np.arange(240*135).reshape(240,135)
    warped,_=deep_view.warp_map(coarse,origin,origin,8)
    assert np.array_equal(warped,np.repeat(np.repeat(coarse,8,axis=0),8,axis=1))


def test_scheduler_replaces_unstarted_work_reuses_results_and_refines():
    worker=FakeTransport()
    scheduler=DeepController(worker)
    high=hp.from_strings('1.23456789123456789123456789','.2',1e-40)
    assert scheduler.tick(high,True) is None
    b=hp.apply_drag(high,20,0)
    c=hp.apply_drag(b,20,0)
    scheduler.tick(b,True)
    scheduler.tick(c,True)
    assert len(worker.submitted)==1
    assert any(r['event']=='cancel' and r['x0']==deep_view.state_key(b)[0] for r in scheduler.records)
    worker.complete()
    source=scheduler.tick(c,True)
    assert source['stride']==8 and len(worker.submitted)==2
    assert worker.submitted[-1]['x0']==deep_view.state_key(c)[0]
    worker.complete()
    scheduler.tick(c,True)
    assert len(worker.submitted)==2  # no full map while moving
    scheduler.tick(c,False)
    assert worker.submitted[-1]['stride']==1
    worker.complete()
    for _ in range(30):
        scheduler.tick(c,False)
    assert len(worker.submitted)==3 and scheduler.final_for(c)['map'].shape==(1920,1080)
    units=[scheduler._unit(r) for r in worker.submitted]
    assert len(units)==len(set(units))
    scheduler.tick(high,False)  # cached coarse, final is the only missing stage
    assert worker.submitted[-1]['stride']==1


def test_actual_frame_orchestration_presents_during_delayed_work_and_holds_final(monkeypatch):
    v=fake_viewer()
    v.frame('initial',10.)
    worker=v.deep_controller.worker
    assert v.frames[-1]['present_kind']=='deep_warped'
    v.on_input(dict(type='drag',dx=20,dy=0))
    t=v.last_input_t
    worker.complete()
    v.frame('fast',t+.01)
    assert v.frames[-1]['present_kind']=='deep_warped'
    worker.complete()
    v.frame('fast',t+.1)
    v.frame('fast',t+.3)
    assert v.frames[-1]['underlying_kind']=='deep_coarse'
    assert worker.submitted[-1]['stride']==1
    worker.complete()
    v.frame('fast',t+.4)
    assert v.frames[-1]['present_kind']=='deep_final' and not v.frames[-1]['intermediate']
    n=len(worker.submitted)
    for i in range(10):
        v.frame('fast',t+.5+i*.01)
        assert v.frames[-1]['present_kind']=='deep_hold' and not v.frames[-1]['intermediate']
    assert len(worker.submitted)==n and v.disp.presents==len(v.frames)
    assert all(f['compute_path']=='perturb' and f['compute_ms']==0 for f in v.frames)


def test_spawn_real_process_with_fake_cpu_engine_and_plan_reuse(tmp_path):
    worker=ProcessWorker(tmp_path,cpu_engine_factory)
    events=[]
    try:
        base=dict(x0='1.2',y0='.3',eps=1e-40,eps_hex=(1e-40).hex(),max_iter=1040,last_input=-1)
        for ident,stride in ((0,8),(1,1),(2,1)):
            request=dict(base,id=ident,stride=stride)
            if ident:
                request.update(x0='1.2000',y0='3.00E-1')
            worker.submit(request)
            deadline=time.perf_counter()+10
            while time.perf_counter()<deadline:
                events.extend(worker.poll())
                if any(r.get('id')==ident and r['event'] in ('end','duplicate_rejected') for r in events):
                    break
                time.sleep(.005)
            else:
                raise AssertionError('CPU spawn engine timeout')
        results=[r for r in events if r['event']=='end']
        assert [r['stride'] for r in results]==[8,1]
        assert [r['info']['plan_index'] for r in results]==[1,1]
        assert len([r for r in events if r['event']=='duplicate_rejected'])==1
        ready=next(r for r in events if r['event']=='fake_ready')
        assert ready['pid']!=os.getpid() and not ready['cupy_imported']
        assert all(r['t_end']>=r['t_start'] for r in results)
    finally:
        worker.close()


def test_replay_thread_queues_independently_and_latency_uses_scheduled_time():
    t0=time.perf_counter()
    replay=deep_view.ThreadedReplay([dict(t=i*.016,type='drag',dx=20,dy=0) for i in range(4)],t0).start()
    try:
        replay.thread.join(1.)  # renderer deliberately does no work until producer finishes
        events=list(replay.drain())
    finally:
        replay.close()
    assert [e['input_index'] for e in events]==list(range(4))
    assert [e['t_scheduled'] for e in events]==[t0+i*.016 for i in range(4)]
    assert all(e['t_enqueued']>=e['t_scheduled'] for e in events)
    frames=[dict(t_present=t0-.01,last_input=-1),dict(t_present=t0+.05,last_input=2),dict(t_present=t0+.08,last_input=3)]
    result=deep_view.timing_metrics(events,frames,t0,t0+.08)
    assert np.allclose(result['input_latency_ms']['values_ms'],[50,34,18,32])
    assert np.allclose(result['present_interval_ms']['values_ms'],[60,30])
    assert result['input_latency_ms']['p95']==pytest.approx(np.percentile([50,34,18,32],95),abs=1e-4)


def test_scripts_and_validation_reject_bad_events():
    scripts=generate(position())
    for name,data in scripts.items():
        result=validate(data,crossing='crossing' in name)
        assert result['min_eps']>=1e-40
        assert data['event_count']==len(data['events'])
        assert data['duration']>data['events'][-1]['t']
        checked=json.loads((Path(app.HERE)/'scripts'/(name+'.json')).read_text(encoding='utf-8'))
        assert checked['events']==data['events']
    cross=scripts['deepzoom2_crossing']
    assert cross['crossing_count']>=6 and cross['drag_count']>=50
    for mutation in ('spacing','drag','wheel','limit'):
        bad=copy.deepcopy(cross)
        if mutation=='spacing': bad['events'][1]['t']=bad['events'][0]['t']+.001
        if mutation=='drag': bad['events'][0]['dx']=19
        if mutation=='wheel': next(e for e in bad['events'] if e['type']=='wheel')['steps']=2
        if mutation=='limit': bad['start'][2]=1e-40
        with pytest.raises(ValueError): validate(bad,crossing=True)


def test_crossing_alignment_scan_and_frame_summary():
    from scripts.deepzoom2_crossing import scan_offsets,frame_summary
    x,y=np.indices((1920,1080))
    m=((x//19+y//23)%2).astype(bool)
    r=scan_offsets(m,m,['1.123456789123456789','.2'])
    assert r['max_offset']==[0,0] and r['ties']==[[0,0]] and r['counted']
    assert r['zero_in_ties'] and len(r['scan'])==121
    shifted=np.roll(m,2,axis=0)
    r=scan_offsets(m,shifted,['1.123456789123456789','.2'])
    assert r['max_offset']==[-2,0]
    v=fake_viewer()
    v.frame('initial',10.)
    summary=frame_summary(v.frames)
    assert summary['excess_axis_count']==0 and summary['frame_count']==1


def test_pure_modules_do_not_import_cuda_in_fresh_interpreter():
    import subprocess
    code="import sys; import viewer.deep_worker, viewer.deep_view, viewer.deepzoom2_scripts, scripts.deepzoom2_viewer_measure, scripts.deepzoom2_crossing; print(int('cupy' in sys.modules or 'mytet_cuda.gpu' in sys.modules))"
    out=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True).stdout.strip()
    assert out=='0'


def test_real_deep_loop_keeps_presenting_while_fake_engine_is_busy(monkeypatch):
    v=fake_viewer()
    now=[100.]
    events=[dict(t=.016*k,type='drag',dx=20 if k<3 else -20,dy=0) for k in range(6)]
    monkeypatch.setattr(app.time,'perf_counter',lambda:now[0])

    class Delayed(FakeTransport):
        def submit(self,r):
            super().submit(r)
            self.due=now[0]+.08

        def poll(self):
            if self.active is not None and now[0]>=self.due:
                self.complete()
            return super().poll()

    class Replay:
        def __init__(self): self.k=0
        def drain(self):
            while self.k<len(events) and now[0]>=100+events[self.k]['t']:
                ev=dict(events[self.k],t_scheduled=100+events[self.k]['t'],t_enqueued=100+events[self.k]['t'])
                self.k+=1
                yield ev
    v.deep_controller=DeepController(Delayed())
    v.glfw=SimpleNamespace(poll_events=lambda:None,window_should_close=lambda w:False)
    v.disp.win=None
    def present():
        now[0]+=.013
        assert now[0]<102,'loop failed to reach final map'
        return 0
    v.disp.present=present
    v.replay=Replay()
    v.meta=dict(script_duration=.5)
    v.frame('initial',now[0])
    v._loop(events,None,None,100.,len(events))
    assert v._k==len(events)
    assert len(v.frames)>30 and v.deep_controller.final_for(v.hp) is not None
    metrics=deep_view.timing_metrics(v.input_log,v.frames,100.,v.frames[-1]['t_present'])
    assert metrics['present_interval_ms']['max']==pytest.approx(13.)
    assert metrics['input_latency_ms']['max']<26.1
    assert sum(f['underlying_kind']=='deep_warped' for f in v.frames)>6
    assert any(f['underlying_kind']=='deep_coarse' for f in v.frames)
    assert any(not f['intermediate'] for f in v.frames)


def test_save_final_raw_maps_and_measurement_definitions(tmp_path):
    from scripts.deepzoom2_viewer_measure import summarize,write_report
    v=fake_viewer()
    initial=v.hp
    v.frame('initial')
    ev=dict(type='drag',dx=20,dy=0,t=1.,i=0)
    v.on_input(dict(ev,input_index=0,t_scheduled=1.,t_enqueued=1.005))
    v.frame('fast')
    w=v.deep_controller.worker
    w.complete()
    v.last_input_t=time.perf_counter()-1
    v.frame('precise')  # retire initial-state coarse, launch input-state coarse
    w.complete()
    v.frame('precise')
    w.complete()
    v.frame('precise')
    v.log_dir=tmp_path
    v.fresh=[]
    v.fresh_bits=[]
    v.recorded=[]
    v.media=[]
    v.job=None
    script=tmp_path/'script.json'
    script.write_text(json.dumps(dict(start=list(map(float,initial)),start_hp=list(map(str,initial[:2])),
                                     events=[ev],duration=1.)),encoding='utf-8')
    v.meta=dict(mode='threaded',stop_delay=.25,t_end=v.frames[-1]['t_present'],script=str(script),
                script_duration=1.,events_delivered=1,t_playback_start=0.)
    v.save()
    with np.load(tmp_path/'deep_final_maps.npz') as archive:
        final=next(iter(v.deep_controller.finals.values()))
        assert np.array_equal(archive[str(final['id'])],np.frombuffer(final['bits'],np.uint8))
    rows=[json.loads(s) for s in (tmp_path/'deep_final_maps.jsonl').read_text().splitlines()]
    assert rows[0]['x0']==deep_view.state_key(v.hp)[0] and rows[0]['eps_hex']==v.hp[2].hex()
    (tmp_path/'contamination.json').write_text(json.dumps({'verdict':{'contaminated':False}}))
    r=summarize(tmp_path)
    assert r['duplicate_jobs']==0 and r['final_picture_count']==1
    assert r['final_after_last_input'] and not r['missing_inputs']
    write_report(tmp_path,r)
    assert (tmp_path/'measure.md').exists()
    contamination={'verdict':{'contaminated':True}}
    (tmp_path/'contamination.json').write_text(json.dumps(contamination))
    with pytest.raises(ValueError,match='contaminated'): summarize(tmp_path)
    v._archive().close(0.)


def test_state_key_exact_decimal_equivalence_without_context_rounding():
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec=3
        for a,b in [('1.2000','12E-1'),('-0.000','0E+7'),('1000.00','1E3')]:
            assert deep_view.state_key((a,b,1e-40))==deep_view.state_key((b,a,1e-40))
        a='1.23456789012345678901234567890123456789'
        b='1.23456789012345678901234567890123456788'
        assert deep_view.state_key((a,'0',1e-40))!=deep_view.state_key((b,'0',1e-40))
        assert deep_view.state_key((a,'0',1e-40))[0]==a


def test_roundtrip_drag_reuses_coarse_and_final_with_no_recomputation():
    point=position()
    high=hp.from_strings(point['x0'],point['y0'],1e-40)
    returned=hp.apply_drag(hp.apply_drag(high,20,0),-20,0)
    assert high==returned and str(high[0])!=str(returned[0])
    w=FakeTransport()
    controller=DeepController(w)
    controller.tick(high,False)
    w.complete()
    coarse=controller.tick(high,True)
    assert coarse['stride']==8
    controller.tick(hp.apply_drag(high,20,0),True)
    assert controller.tick(returned,True) is coarse
    w.complete()
    assert controller.tick(returned,True) is coarse
    controller.tick(returned,False)
    w.complete()
    final=controller.tick(returned,False)
    assert final['stride']==1
    before=len(w.submitted)
    controller.tick(hp.apply_drag(high,20,0),True)  # cached moved coarse
    for moving in (True,False,False):
        assert controller.tick(returned,moving) is final
    assert controller.final_for(high) is controller.final_for(returned) is final
    assert len(w.submitted)-before==0


def measurement_fixture(tmp_path):
    """Full independent CPU playback and synthetic presents, never GPU evidence."""
    script=generate(position())['deepzoom2_responsive']
    script_path=tmp_path/'original.json'
    script_path.write_text(json.dumps(script),encoding='utf-8')
    state,high=deep_state.initial_states(script['start'],script['start_hp'])
    inputs,frames=[],[]
    for k,ev in enumerate(script['events']):
        before=high
        state,high,clamped=deep_state.apply_event(state,high,ev)
        inputs.append(dict(ev,input_index=k,t_scheduled=100+ev['t'],t_processed=100+ev['t']+.001,
                           before_hp=hp.to_record(before),after_hp=hp.to_record(high),clamped=clamped))
        frames.append(dict(seq=k,last_input=k,t_present=100+ev['t']+.01,intermediate=True))
    key=deep_view.state_key(high)
    request=dict(id=7,x0=key[0],y0=key[1],eps_hex=key[2],stride=1)
    frames.append(dict(seq=len(frames),last_input=len(inputs)-1,t_present=100+script['duration']+.01,
                       underlying_kind='deep_final',intermediate=False,source_request_id=7,
                       source_stride=1,source_hp=hp.to_record(high),deep=request))
    jobs=[dict(request,event='start'),dict(request,event='end',t_end=111.,t_received=111.001,
                                         info=dict(out_of_range=False))]
    meta=dict(mode='threaded',script=str(script_path),script_duration=script['duration'],
              t_playback_start=100.,events_delivered=len(inputs),t_end=frames[-1]['t_present'],stop_delay=.25)
    data={'meta.json':meta,'inputs.jsonl':inputs,'frames.jsonl':frames,'deep_jobs.jsonl':jobs,
          'contamination.json':dict(verdict=dict(contaminated=False))}
    return data


def write_measurement_fixture(tmp_path,data):
    for name,rows in data.items():
        content='\n'.join(map(json.dumps,rows)) if name.endswith('.jsonl') else json.dumps(rows)
        (tmp_path/name).write_text(content,encoding='utf-8')


def test_measurement_counts_numerically_equal_start_twice_as_duplicate(tmp_path):
    from scripts.deepzoom2_viewer_measure import summarize
    data=measurement_fixture(tmp_path)
    write_measurement_fixture(tmp_path,data)
    assert summarize(tmp_path)['duplicate_jobs']==0
    duplicate=dict(data['deep_jobs.jsonl'][0],id=8)
    duplicate['x0']=str(hp.D(duplicate['x0']))+'0'
    duplicate['y0']=str(hp.D(duplicate['y0']))+'0'
    data['deep_jobs.jsonl'].append(duplicate)
    write_measurement_fixture(tmp_path,data)
    assert summarize(tmp_path)['duplicate_jobs']==1  # deliberately duplicated computation
    data['deep_jobs.jsonl'].append(duplicate.copy())
    write_measurement_fixture(tmp_path,data)
    assert summarize(tmp_path)['duplicate_jobs']==2  # every extra start counts, even with a reused id
    data['deep_jobs.jsonl'].pop()
    duplicate['stride']=8
    write_measurement_fixture(tmp_path,data)
    assert summarize(tmp_path)['duplicate_jobs']==0  # a different stage is not a duplicate


@pytest.mark.parametrize('mutation',['prefix','empty','index','number','kind','value','schedule',
                                    'duration','present_duration','target','stride','intermediate','delivered_count'])
def test_measurement_rejects_incomplete_or_changed_script(tmp_path,mutation):
    from scripts.deepzoom2_viewer_measure import summarize,write_report
    data=measurement_fixture(tmp_path)
    inputs,frames,meta=data['inputs.jsonl'],data['frames.jsonl'],data['meta.json']
    write_measurement_fixture(tmp_path,data)
    assert summarize(tmp_path)['valid']
    if mutation in ('prefix','empty'):
        data['inputs.jsonl']=inputs[:1] if mutation=='prefix' else []
        meta['events_delivered']=len(data['inputs.jsonl'])
        frames[-1].update(last_input=0,source_hp=inputs[0]['after_hp'])
        frames[-1]['deep']=dict(x0=inputs[0]['after_hp'][0],y0=inputs[0]['after_hp'][1],eps_hex=inputs[0]['after_hp'][3])
    elif mutation=='index': inputs[1]['input_index']=0
    elif mutation=='number': inputs[1]['i']=99
    elif mutation=='kind': inputs[1]['type']='wheel'
    elif mutation=='value': inputs[1]['dx']+=1
    elif mutation=='schedule': inputs[1]['t_scheduled']+=.01
    elif mutation=='duration': meta['t_end']=100+meta['script_duration']-.01
    elif mutation=='present_duration': frames[-1]['t_present']=100+meta['script_duration']-.01
    elif mutation=='target': frames[-1]['deep']=dict(frames[-1]['deep'],x0='0')
    elif mutation=='stride': frames[-1]['source_stride']=8
    elif mutation=='intermediate': frames[-1]['intermediate']=True
    elif mutation=='delivered_count': meta['events_delivered']-=1
    write_measurement_fixture(tmp_path,data)
    r=summarize(tmp_path)
    assert r['valid'] is False and r['status']=='invalid' and r['measurement_failure']
    assert 'input_latency_ms' not in r
    write_report(tmp_path,r)
    assert json.loads((tmp_path/'measure.json').read_text())['valid'] is False
    assert r['measurement_failure'] in (tmp_path/'measure.md').read_text(encoding='utf-8')


def test_window_assertions_reject_render_inversion_offsets_ties_and_no_counted_crossings():
    import runpy
    window=runpy.run_path(str(Path(__file__).with_name('test_deepzoom2_viewer_window.py')))
    m=np.zeros((1920,1080),bool)
    m[14:31,72:90]=True
    rt=np.empty((1080,1920,4),np.uint8)
    rt[...,:3]=(m.T.astype(np.uint8)*255)[...,None]
    rt[...,3]=255
    assert window['final_render_sha'](np.packbits(m))==hashlib.sha256(rt.tobytes()).hexdigest()
    frames=[dict(seq=0,underlying_kind='deep_final',intermediate=False,source_stride=1,
                 source_request_id=7,rt_sha256=hashlib.sha256(rt.tobytes()).hexdigest(),
                 deep=dict(x0='1.2000',y0='-0.0'),display_params=[1.2,0.,1e-40],
                 source_hp=['12E-1','0',1e-40,(1e-40).hex()])]
    expected={7:window['final_render_sha'](np.packbits(m))}
    window['assert_final_targets'](frames,expected)
    frames[0]['rt_sha256']=window['final_render_sha'](np.packbits(~m))
    with pytest.raises(AssertionError): window['assert_final_targets'](frames,expected)
    r=dict(map_comparison_computed=True,frame_count=1,excess_axis_count=0,
           frames=[dict(center_diff_px=[1.,3.],f64_round_limit_px=[.5,3.])],counted_crossings=6,
           crossings=[dict(counted=True,max_offset=[0,0],ties=[[0,0]]) for _ in range(6)])
    window['assert_crossing_result'](r)
    for change in ('offset','tie','count','center'):
        bad=copy.deepcopy(r)
        if change=='offset': bad['crossings'][0]['max_offset']=[2,0]
        if change=='tie': bad['crossings'][0]['ties']=[[1,0],[-1,0]]
        if change=='count':
            bad['counted_crossings']=0
            for row in bad['crossings']: row['counted']=False
        if change=='center': bad['frames'][0]['center_diff_px'][0]=1.01
        with pytest.raises(AssertionError): window['assert_crossing_result'](bad)


SCRIPT_DIR = Path(app.HERE)/'scripts'
LEGACY_SCRIPTS = [SCRIPT_DIR/n for n in ('D1.json','Z1.json')] + sorted(SCRIPT_DIR.glob('P_*.json'))
SCENES = json.loads((SCRIPT_DIR/'deepzoom_scenes.json').read_text(encoding='utf-8'))['scenes']
LEGACY_SCRIPTS += [SCRIPT_DIR/f'{scene}_{kind}.json' for scene in SCENES for kind in ('manip','scroll')]


@pytest.mark.parametrize('path',LEGACY_SCRIPTS,ids=lambda p:p.stem)
def test_actual_input_preserves_legacy_float_contract_bit_for_bit(path):
    from tests.viewer_check import contract_step, expected_params
    data=json.loads(path.read_text(encoding='utf-8'))
    v=fake_viewer()
    v.state,v.hp=deep_state.initial_states(data['start'])
    expected=tuple(map(float,data['start']))
    for ev in data['events']:
        expected=contract_step(expected,ev)
        v.on_input(ev)
        assert tuple(x.hex() for x in v.state)==tuple(x.hex() for x in expected)
        assert v.state[2].hex()==v.hp[2].hex()
        assert v.input_log[-1]['resync'] is False
    assert v.state==expected_params(data,None)
    assert sum(r['resync'] for r in v.input_log)==0


@pytest.mark.parametrize('kind',['crossing','drags'])
def test_actual_input_resyncs_near_crossing_and_records_decimal_drift(kind,tmp_path):
    from scripts.deepzoom2_crossing import analyze
    v=fake_viewer(1.01e-13)
    if kind=='crossing':
        data=json.loads((SCRIPT_DIR/'deepzoom2_crossing.json').read_text(encoding='utf-8'))
        v.state,v.hp=deep_state.initial_states(data['start'],data['start_hp'])
        events=data['events']
    else:
        events=[dict(type='drag',dx=20,dy=20*(-1 if k%3 else 1)) for k in range(600)]
    frames=[]
    for k,ev in enumerate(events):
        v.on_input(ev)
        diag=deep_state.diagnostics(v.state,v.hp)
        assert all(abs(d)<=max(1,b) for d,b in zip(diag['center_diff_px'],diag['f64_round_limit_px']))
        assert all(abs(d)<=max(.5,b) for d,b in zip(diag['center_diff_px'],diag['f64_round_limit_px']))
        assert v.state[2].hex()==v.hp[2].hex()
        frames.append(dict(diag,seq=k,display_params=v.state))
    count=sum(r['resync'] for r in v.input_log)
    assert count>0
    for row in v.input_log:
        if row['resync']:
            assert any(abs(hp.D(d))>hp.D('.5') for d in row['pre_resync_center_diff_px'])
    (tmp_path/'inputs.jsonl').write_text('\n'.join(map(json.dumps,v.input_log)),encoding='utf-8')
    (tmp_path/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)),encoding='utf-8')
    result=analyze(tmp_path,tmp_path,maps=False)
    assert result['resync_count']==count and result['excess_axis_count']==0
    assert f'|재동기 횟수|{count}|' in (tmp_path/'crossing.md').read_text(encoding='utf-8')


def settle(controller, high):
    for _ in range(4):
        controller.tick(high,False)
        if controller.worker.active is not None:
            controller.worker.complete()
    return controller.final_for(high)


def test_bounded_map_and_packed_lru_stream_all_finals_before_save(tmp_path):
    controller=DeepController(FakeTransport(),tmp_path)
    highs=[hp.from_strings(str(k),'.2',1e-40) for k in range(258)]
    first=settle(controller,highs[0])
    for high in highs[1:16]:
        settle(controller,high)
    assert len(controller.cache)==16
    # Decoded eviction restores from packed bits with no additional work.
    before=len(controller.worker.submitted)
    assert controller.final_for(highs[0])['id']==first['id']
    assert len(controller.worker.submitted)==before
    for high in highs[16:]:
        settle(controller,high)
        assert len(controller.cache)<=16 and len(controller.finals)<=256 and len(controller.coarse)<=256
        assert all('map' not in row for row in controller.finals.values())
    assert len(controller.computed)==516
    assert controller.final_for(highs[1]) is None
    assert controller.final_for(highs[0])['id']==first['id']  # recently used packed result survives
    # Raw records survive memory eviction immediately, before Viewer.save/close.
    with np.load(tmp_path/'deep_final_maps.npz') as archive:
        assert len(archive.files)==258
        assert np.array_equal(archive[str(first['id'])],np.packbits(first['map']))
    rows=[json.loads(s) for s in (tmp_path/'deep_final_maps.jsonl').read_text().splitlines()]
    assert len(rows)==258 and rows[0]['id']==first['id']
    settle(controller,highs[1])
    recomputed=[r for r in controller.records if r['event']=='recompute_after_eviction']
    assert [r['stride'] for r in recomputed]==[8,1]
    assert all(r['recompute_after_eviction'] for r in recomputed)
    controller.save_final_archive(tmp_path)
    with np.load(tmp_path/'deep_final_maps.npz') as archive:
        assert len(archive.files)==259
    assert len(controller.finals)==256


@pytest.mark.parametrize('name',['deepzoom2_responsive','deepzoom2_crossing','deepzoom2_video'])
def test_script_playback_never_recomputes_after_eviction(name):
    data=json.loads((SCRIPT_DIR/(name+'.json')).read_text(encoding='utf-8'))
    state,high=deep_state.initial_states(data['start'],data.get('start_hp'))
    controller=DeepController(FakeTransport())
    for k,ev in enumerate(data['events']):
        state,high,clamped=deep_state.apply_event(state,high,ev)
        assert not clamped
        controller.tick(high,True)
        if controller.worker.active is not None:
            controller.worker.complete()
        controller.tick(high,True)
        next_t=data['events'][k+1]['t'] if k+1<len(data['events']) else data['duration']
        if next_t-ev['t']>=app.STOP_DELAY:
            settle(controller,high)
    settle(controller,high)
    units=[controller._unit(r) for r in controller.worker.submitted]
    assert len(units)==len(set(units))
    assert not any(r['event']=='recompute_after_eviction' for r in controller.records)


def test_child_plan_lru_evicts_oldest_and_reuses_recent_without_cuda():
    import queue
    from viewer.deep_worker import worker_main
    requests,responses=queue.Queue(),queue.Queue()
    specs=[(k,8) for k in range(8)]+[(0,1),(8,8),(1,1)]
    for ident,(key,stride) in enumerate(specs):
        requests.put(dict(id=ident,x0=str(key),y0='.2',eps_hex=(1e-40).hex(),stride=stride,max_iter=1040))
    requests.put(dict(id=len(specs),x0='1',y0='.2',eps_hex=(1e-40).hex(),stride=1,max_iter=1040,
                      recompute_after_eviction=True))
    requests.put(None)
    worker_main(requests,responses,'.',cpu_engine_factory)
    rows=[]
    while not responses.empty(): rows.append(responses.get_nowait())
    assert not any(r['event']=='error' for r in rows)
    ends=[r for r in rows if r['event']=='end']
    assert [r['info']['plan_index'] for r in ends]==list(range(1,9))+[1,9,10,10]
    assert ends[-1]['recompute_after_eviction'] is True


def test_measurement_counts_eviction_marker_once_even_without_first_start(tmp_path):
    from scripts.deepzoom2_viewer_measure import summarize
    data=measurement_fixture(tmp_path)
    start=data['deep_jobs.jsonl'][0]
    data['deep_jobs.jsonl'].append(dict(start,event='recompute_after_eviction',recompute_after_eviction=True))
    start['recompute_after_eviction']=True
    write_measurement_fixture(tmp_path,data)
    result=summarize(tmp_path)
    assert result['duplicate_jobs']==result['recompute_after_eviction']==1
    data['deep_jobs.jsonl'].insert(0,dict(start,id=1,recompute_after_eviction=False))
    write_measurement_fixture(tmp_path,data)
    result=summarize(tmp_path)
    assert result['duplicate_jobs']==result['recompute_after_eviction']==1


@pytest.mark.parametrize('delta,resync',[('-1e-60',False),('1e-60',True)])
def test_resync_threshold_uses_decimal_without_ambient_abs_rounding(delta,resync):
    from decimal import localcontext
    state=(0.,0.,1.)
    drift=hp.CTX.add(hp.D('.5'),hp.D(delta))
    x=hp.CTX.multiply(hp.spacing(1.)[0],drift).copy_negate()
    high=(x,hp.D(0),1.)
    record={}
    with localcontext() as ctx:
        ctx.prec=3
        updated,updated_hp,clamped=deep_state.apply_event(state,high,dict(type='wheel',steps=0),record)
    assert record['resync'] is resync
    assert updated_hp==high and not clamped
    assert updated==(deep_state.project(high) if resync else state)


def test_interactive_deep_engine_skips_the_free_memory_rule(monkeypatch,tmp_path):
    """The FreeVirtualMemory rule protects test, measurement and recording runs. The interactive viewer's deep child
    must not apply it: with less free memory, a deep zoom would otherwise end the viewer."""
    import queue
    import types
    import deep
    import mytet_cuda
    from viewer import deep_worker
    calls=[]
    def require_memory():
        calls.append('require_memory')
        raise RuntimeError('FreeVirtualMemory 1 KB < 8388608 KB')
    tools=types.ModuleType('scripts.perturb_tools')
    tools.require_memory=require_memory
    tools.audit_gpu=lambda stage,directory:dict(stage=stage)
    gpu=types.ModuleType('mytet_cuda.gpu')
    gpu.device_info=lambda:dict(name='fake')
    gpu.cp=SimpleNamespace(empty=lambda n:None)
    perturb=types.ModuleType('deep.perturb')
    monkeypatch.setitem(sys.modules,'scripts.perturb_tools',tools)
    for package,name,module in ((mytet_cuda,'gpu',gpu),(deep,'perturb',perturb)):
        monkeypatch.setitem(sys.modules,module.__name__,module)
        monkeypatch.setattr(package,name,module,raising=False)
    assert deep_worker.engine_factory_for('interactive') is deep_worker.interactive_gpu_engine
    assert all(deep_worker.engine_factory_for(m) is None
               for m in ('timed','verify','media','callbacks','honesty','threaded'))
    responses=queue.Queue()
    assert deep_worker.interactive_gpu_engine(tmp_path,responses) is perturb
    assert calls==[] and responses.get_nowait()['event']=='audit'
    with pytest.raises(RuntimeError,match='FreeVirtualMemory'):
        deep_worker.gpu_engine(tmp_path,responses)
    assert calls==['require_memory']


@pytest.mark.parametrize('mode,factory',[('interactive','interactive_gpu_engine'),('threaded',None),('timed',None)])
def test_viewer_mode_picks_the_deep_engine(monkeypatch,mode,factory):
    from viewer import deep_worker
    class Started(Exception):
        pass
    seen=[]
    def recorder(directory,engine_factory=None):
        seen.append(engine_factory)
        raise Started
    monkeypatch.setattr(deep_worker,'ProcessWorker',recorder)
    v=fake_viewer()
    v.mode,v.deep_controller=mode,None
    with pytest.raises(Started):
        v._deep_frame('fresh',v.state,0.)
    assert seen==[getattr(deep_worker,factory) if factory else None]
