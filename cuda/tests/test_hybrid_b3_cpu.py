"""B3 controls: real production plans, independent record checks; no window/CUDA."""
import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

F=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(F/"viewer"),str(F/"scripts")]
import hybrid
import viewer_check as vc
from deferred import MapArchive
from margins import MarginScheduler
import cpu_guard


def times(record, t):
    for prefix in ("sha","pack"):
        record[prefix+"_t_start"]=t
        record[prefix+"_t_end"]=t+.001
        record[prefix+"_thread"]={"name":"map-archive_0","ident":123}


def recording():
    p=(0.,0.,1.)
    screen=np.zeros((1920,1080),bool)
    extended=np.ones((1924,1082),bool)
    extended[2:1922,1:1081]=screen
    fresh={0:{"id":0,"params":p,"margin_width":[2,1],"margin_t_end":2.5}}
    fresh[0].update(margin_shape=[1924,1082],margin_chunks=[
        {"rect":list(rect),"t_launch":2.,"t_complete":2.5} for rect in
        ((0,2,0,1082),(1922,1924,0,1082),(2,1922,0,1),(2,1922,1081,1082))])
    times(fresh[0],2.)
    initial=dict(seq=0,kind="initial",present_kind="initial",precise_id="precise_0",display_params=p,
                 t_decide=.5,t_present=1.,source_fresh_id=None,bound_source_fresh_id=None,scale=1.,offset=[0.,0.])
    times(initial,1.1)
    tr={"id":"precise_0","scale":1.,"offset":[0.,0.]}
    initial.update(candidate_ids=["precise_0"],bound_candidate_ids=["precise_0"],candidate_transforms=[tr],
                   candidates=[dict(id="precise_0",part="precise",transform=tr)],
                   pixel_counts=dict(newest=0,older=0,margin=0,precise=2073600,edge=0))
    frames=[initial]
    for seq,display,decide,present,part in ((1,p,1.1,1.5,"screen"),(2,(2/1919,0.,1.),3.,3.1,"screen+margin")):
        src=dict(fresh[0],part=part,margin=[2,1] if part=="screen+margin" else [0,0])
        sources=[src,{"id":"precise_0","params":p,"part":"precise","margin":[0,0]}]
        _,transforms,counts=hybrid.candidate_plan3(display,sources)
        frames.append(dict(seq=seq,kind="fast",present_kind="warped",display_params=display,
                           t_decide=decide,t_present=present,source_fresh_id=0,bound_source_fresh_id=0,
                           candidates=[dict(id=r["id"],part=r["part"],transform=tr) for r,tr in zip(sources,transforms)],
                           candidate_ids=[0,"precise_0"],candidate_transforms=transforms,
                           bound_candidate_ids=[0,"precise_0"],pixel_counts=counts,
                           scale=transforms[0]["scale"],offset=transforms[0]["offset"]))
    return frames,fresh,{0:screen},{0:extended},{"precise_0":screen}


def test_margin_control_normal_then_deliberately_violating_record():
    frames,fresh,*_=recording()
    assert vc.b3_record_checks(frames,fresh)["margin_ignored"]==[], "SIG:margin_ignored normal"
    assert not any(vc.validate_b3(frames,fresh).values())
    bad=copy.deepcopy(frames)
    bad[-1]["candidates"][0]["part"]="screen"
    _,bad[-1]["pixel_counts"]=vc.independent_b3(bad[-1],fresh,precise_params={"precise_0":frames[0]["display_params"]})
    assert vc.b3_record_checks(bad,fresh)["margin_ignored"]==[2], "SIG:margin_ignored violating control"


@pytest.mark.parametrize("prefix",["sha","pack"])
def test_sha_control_normal_then_deliberately_violating_record(prefix):
    frames,fresh,*_=recording()
    assert vc.b3_record_checks(frames,fresh)["sha_on_path"]==[], "SIG:sha_on_path normal"
    bad=copy.deepcopy(fresh)
    bad[0][prefix+"_t_start"]=frames[1]["t_present"]-.001
    assert vc.b3_record_checks(frames,bad)["sha_on_path"]==[0], "SIG:sha_on_path violating control"


def test_live_margin_record():
    frames,fresh,maps,margins,precise=recording()
    f=frames[-1]
    candidates=hybrid.motion_candidates([(1,fresh[0],maps[0])],
                ({"id":"precise_0","params":frames[0]["display_params"]},precise["precise_0"]),
                f["t_decide"],margins.get)
    _,transforms,counts=hybrid.candidate_plan3(f["display_params"],[r for r,_ in candidates])
    f["candidates"]=[dict(id=r["id"],part=r["part"],transform=tr) for (r,_),tr in zip(candidates,transforms)]
    f["pixel_counts"]=counts
    assert vc.b3_record_checks(frames,fresh)["margin_ignored"]==[], "SIG:margin_ignored"


def test_live_sha_record(monkeypatch):
    from test_hybrid_cpu import b2_cpu_viewer
    v,clock=b2_cpu_viewer(monkeypatch,.01)
    v.hybrid_version=3
    v.last_precise=None
    v.margin_maps={}
    v.margins=MarginScheduler(None,[],0,0)
    def present():
        clock[0]+=.013
        return 0
    v.disp.present=present
    try:
        v.frame("fast",clock[0])
        v._finish_archive()
        result=vc.b3_record_checks(v.frames,{r["id"]:r for r in v.fresh})
        assert result["sha_on_path"]==[], "SIG:sha_on_path"
    finally:
        v.map_archive.close(clock[0])


def test_background_archive_after_show_unshown_tail_and_failure():
    import time
    archive=MapArchive()
    shown,unshown={},{}
    m=np.zeros((10,8),bool)
    archive.register(("fresh",0),m,shown)
    archive.register(("fresh",1),m.copy(),unshown)
    assert not m.flags.writeable and shown["sha256"] is None
    first=time.perf_counter()
    archive.shown(("fresh",0),first)
    last=time.perf_counter()
    archive.close(last)
    assert shown["sha_t_start"]>=first and shown["pack_t_start"]>=first
    assert unshown["sha_t_start"]>=last and unshown["pack_t_start"]>=last
    assert shown["sha_thread"]["ident"]!=threading.get_ident()
    assert np.array_equal(np.unpackbits(archive.bits(("fresh",0)))[:80].reshape(10,8),m)
    broken=MapArchive()
    broken.register(0,m,{})
    broken._pack=lambda *args: (_ for _ in ()).throw(RuntimeError("worker broke"))
    with pytest.raises(RuntimeError,match="worker broke"):
        broken.close(last)


def test_64_candidates_then_precise_order_ties_and_usable_boundary():
    m=np.zeros((1,1),bool)
    shown=[(i//2,{"id":i,"params":(0.,0.,1.),"margin_t_end":10.,"margin_width":[1,1]},m) for i in range(70)]
    precise=({"id":"precise_0","params":(1.,2.,3.)},m)
    cs=hybrid.motion_candidates(shown,precise,10.,lambda i:~m)
    assert len(cs)==65 and [r["id"] for r,_ in cs[:-1]]==list(range(69,5,-1))
    assert cs[-1][0]["params"]==(1.,2.,3.) and all(r["part"]=="screen" for r,_ in cs[:-1])
    cs=hybrid.motion_candidates(shown,precise,10.0001,lambda i:~m)
    assert all(r["part"]=="screen+margin" for r,_ in cs[:-1])


def test_counts_and_independent_margin_pixels():
    frames,fresh,maps,margins,precise=recording()
    for f in frames:
        m,counts=vc.independent_b3(f,fresh,maps,margins,precise,precise_params={"precise_0":frames[0]["display_params"]})
        assert counts==f["pixel_counts"] and sum(counts.values())==2073600
    assert counts["margin"]==1080 and counts["edge"]==0 and m[-1].all() and not m[:-1].any()
    bad=copy.deepcopy(frames)
    bad[-1]["pixel_counts"]["edge"]=1
    assert vc.b3_record_checks(bad,fresh)["margin_ignored"]==[2]


def test_scheduler_yields_four_chunks_and_started_ring_continues(monkeypatch):
    import margins
    clock=[0.]
    monkeypatch.setattr(margins.time,"perf_counter",lambda:clock[0])
    launched=[]
    class Chunk:
        def __init__(self,r,rect):
            self.rect=rect
            self.ready=False
            launched.append((r["id"],self))
        def poll(self):
            a,b,c,d=self.rect
            return np.ones((b-a,d-c),bool) if self.ready else None
    rects=[(i,i+1,0,1) for i in range(8)]
    scheduler=MarginScheduler(Chunk,rects,1,0,mode="gate")
    a={"id":0,"params":(0.,0.,1.)}
    b={"id":1,"params":(0.,0.,1.)}
    screen=np.zeros((1920,1080),bool)
    scheduler.shown(a,screen,1)
    scheduler.tick(True)
    assert not launched
    scheduler.tick(False)
    assert len(launched)==4 and all(i==0 for i,_ in launched)
    scheduler.shown(b,screen,2)
    for _,chunk in launched:
        chunk.ready=True
    clock[0]=1.
    scheduler.tick(True)
    assert len(launched)==4 and all(c["t_complete"]==1. for c in a["margin_chunks"])
    scheduler.tick(False)
    assert len(launched)==8 and all(i==0 for i,_ in launched[4:])
    while b["margin_t_end"] is None:
        for _,chunk in launched:
            chunk.ready=True
        clock[0]+=1
        scheduler.tick(False)
    assert b["margin_t_end"]>=max(c["t_complete"] for c in b["margin_chunks"])
    assert not scheduler.completed(1).flags.writeable
    assert len(scheduler.active)<=4


def sample(cpu=0,pids=()):
    return {"other_cpu":cpu,"ti_processes":[{"pid":p,"name":"proc"} for p in pids]}


def test_guard_parsing_all_core_verdict_and_exact_boundaries():
    text=f"OTHER, 10, x\n{cpu_guard.TI_UUID}, 20, viewer.exe\n{cpu_guard.TI_UUID}, 30, other.exe"
    assert [p["pid"] for p in cpu_guard.parse_ti_processes(text)]==[20,30]
    rows=[sample(50) for _ in range(20)]
    rows[0]=sample(51)
    assert not cpu_guard.verdict(rows)["contaminated"]  # exactly 5% is allowed
    rows[1]=sample(51)
    assert cpu_guard.verdict(rows)["contaminated"]
    assert not cpu_guard.verdict([sample(0,[10,20])],[10],[20])["contaminated"]
    assert cpu_guard.verdict([sample(0,[10,20,30])],[10],[20])["new_ti_pids"]==[30]


def test_guard_wait_clean_resets_continuous_window():
    clock=[0.]
    values=iter([0,60,0,0,0])
    source=SimpleNamespace(sample=lambda owned:sample(next(values)))
    def sleep(dt):
        clock[0]+=dt
    result=cpu_guard.wait_clean(window=3,source=source,sleep=sleep,now=lambda:clock[0])
    assert result["waited_s"]==4 and len(result["reasons"])==1


def test_guarded_retry_keeps_contaminated_directories(monkeypatch,tmp_path):
    monkeypatch.setattr(cpu_guard,"wait_clean",lambda:{"waited_s":120,"reasons":[]})
    calls=[]
    def launch(directory):
        directory.mkdir()
        calls.append(directory)
        (directory/"contamination.json").write_text(json.dumps({"verdict":{"contaminated":len(calls)<3}}))
        return directory
    d=cpu_guard.guarded_run(launch,tmp_path/"run")
    assert len(calls)==3 and (tmp_path/"run_contaminated_1").is_dir() and (tmp_path/"run_contaminated_2").is_dir()
    result=json.loads((d/"contamination.json").read_text())
    assert len(result["previous_attempts"])==2 and not result["verdict"]["contaminated"]



def test_guard_counter_subtracts_child_birth_and_final_cpu(monkeypatch):
    from ctypes import wintypes
    state={"total":100,"busy":50,"ticks":{10:10}}
    closed=[]
    def fill(pointer,value):
        pointer._obj.dwLowDateTime=value&0xffffffff
        pointer._obj.dwHighDateTime=value>>32
    def system(idle,kernel,user):
        fill(idle,state["total"]-state["busy"])
        fill(kernel,state["total"])
        fill(user,0)
        return True
    def process(handle,create,exit_,kernel,user):
        fill(kernel,state["ticks"][handle]); fill(user,0)
        return True
    counter=cpu_guard.WindowsCounters.__new__(cpu_guard.WindowsCounters)
    counter.kernel=SimpleNamespace(GetSystemTimes=system,OpenProcess=lambda flags,inherit,pid:pid,
                                    GetProcessTimes=process,CloseHandle=closed.append)
    counter.last=None;counter.handles={};counter.process_ticks={}
    monkeypatch.setattr(cpu_guard,"ti_processes",lambda:[])
    assert counter.sample({10})["other_cpu"]==0
    counter.track(20)  # newly born child; its first measured ticks all belong to this interval
    state.update(total=200,busy=140,ticks={10:20,20:50})
    assert counter.sample({10,20})["other_cpu"]==30
    state.update(total=300,busy=230,ticks={10:30,20:100})  # final child ticks retained after exit
    assert counter.sample({10,20})["other_cpu"]==30 and closed==[]
    counter.close()
    assert sorted(closed)==[10,20]


def test_auxiliary_archive_sha_and_embedded_screen(tmp_path):
    frames,fresh,maps,margins,precise=recording()
    fresh[0].update(margin_shape=list(margins[0].shape),margin_archive={"sha256":vc.sha(margins[0])})
    frames[0]["map_sha256"]=vc.sha(precise["precise_0"])
    np.save(tmp_path/"fresh_maps.npy",np.packbits(maps[0])[None,:])
    def write():
        np.savez(tmp_path/"fresh_margins.npz",**{"0":np.packbits(margins[0])})
        np.savez(tmp_path/"precise_maps.npz",**{"precise_0":np.packbits(precise["precise_0"])})
    write()
    loaded,pm=vc.load_auxiliary(tmp_path,fresh,frames)
    assert np.array_equal(loaded[0],margins[0]) and np.array_equal(pm["precise_0"],precise["precise_0"])
    margins[0][2,1]=True
    write()
    with pytest.raises(ValueError,match="margin archive"):
        vc.load_auxiliary(tmp_path,fresh,frames)
    fresh[0]["margin_archive"]["sha256"]=vc.sha(margins[0])
    with pytest.raises(ValueError,match="margin archive"):
        vc.load_auxiliary(tmp_path,fresh,frames)
    margins[0][2,1]=False
    fresh[0]["margin_archive"]["sha256"]=vc.sha(margins[0])
    precise["precise_0"]=~precise["precise_0"]
    write()
    with pytest.raises(ValueError,match="precise archive"):
        vc.load_auxiliary(tmp_path,fresh,frames)



def test_next_screen_job_starts_before_draw_without_changing_source(monkeypatch):
    from test_hybrid_cpu import b2_cpu_viewer
    v,clock=b2_cpu_viewer(monkeypatch,.04)
    v.frame("fast",clock[0])
    old_source=v.source["id"]
    calls=[]
    real_start,real_draw=v._start_fresh,v._draw_present
    def start(p,t):
        calls.append("start")
        real_start(p,t)
    def draw(*a,**k):
        calls.append("draw")
        assert v.source["id"]==old_source and v.job is not None
        real_draw(*a,**k)
    v._start_fresh,v._draw_present=start,draw
    clock[0]+=.02
    v.state=(1.,0.,50.)
    v.applied+=1
    v.frame("fast",clock[0])
    assert calls==["start","draw"] and v.frames[-1]["source_fresh_id"]==old_source
    v.map_archive.close(clock[0])


def test_gl_extended_layer_reupload_and_65_candidates_without_context(monkeypatch):
    import render
    uploads,bounds,origins=[],[],[]
    gl=render.GL
    for name in ("glActiveTexture","glBindTexture","glBindFramebuffer","glViewport","glUseProgram",
                 "glUniform1i","glUniform1iv","glDrawArrays"):
        monkeypatch.setattr(gl,name,lambda *a:None)
    monkeypatch.setattr(gl,"glUniform4iv",lambda location,count,data:bounds.append(data.copy()))
    monkeypatch.setattr(gl,"glUniform2iv",lambda location,count,data:origins.append(data.copy()))
    monkeypatch.setattr(gl,"glGetUniformLocation",lambda *a:0)
    monkeypatch.setattr(gl,"glTexSubImage3D",lambda *a:uploads.append(a[-1].shape))
    monkeypatch.setattr(gl,"glTexSubImage2D",lambda *a:None)
    d=render.Display.__new__(render.Display)
    d.source_layers={};d.sources_tex=10;d.multi_indices=[11,12];d.fbo=13;d.p_multi=14
    d._draw_window=lambda:None
    p=(0.,0.,1.);m=np.zeros((1920,1080),bool)
    r=dict(id=0,params=p,part="screen",margin=[0,0])
    d.draw_candidates(p,[(r,m)],version=3)
    extended=np.zeros((1924,1082),bool)
    r=dict(r,part="screen+margin",margin=[2,1])
    d.draw_candidates(p,[(r,extended)],version=3)
    assert uploads==[(1920,1080),(1924,1082)]
    assert bounds[-1].tolist()==[[-2,1922,-1,1081]] and origins[-1].tolist()==[[2,1]]
    candidates=[(dict(id=i,params=p,part="screen",margin=[0,0]),m) for i in range(64)]
    candidates.append((dict(id="precise_0",params=p,part="precise",margin=[0,0]),m))
    d.draw_candidates(p,candidates,version=3)
    assert len(d.source_layers)==65 and d.bound_candidate_ids==list(range(64))+["precise_0"]


def test_extended_axis_and_ring_definition_without_cuda_import():
    import ast
    tree=ast.parse((F/"viewer/fast.py").read_text(encoding="utf-8"))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("extended_axes","margin_rectangles")]
    scope={"np":np,"MX":480,"MY":270,"ROWS_PER_LAUNCH":32}
    exec(compile(ast.Module(body=functions,type_ignores=[]),"axis_contract","exec"),scope)
    for p in ((0.,0.,50.),(-.465,.918,4e-9)):
        for mx,my in ((0,0),(240,135),(480,270),(960,540)):
            x,y=scope["extended_axes"](*p,mx,my)
            assert np.array_equal(x[mx:mx+1920].view(np.uint64),np.linspace(p[0]-p[2],p[0]+p[2],1920).view(np.uint64))
            ey=p[2]*(9/16)
            assert np.array_equal(y[my:my+1080].view(np.uint64),np.linspace(p[1]-ey,p[1]+ey,1080).view(np.uint64))
            cover=np.zeros((1920+2*mx,1080+2*my),np.uint8)
            for a,b,c,d in scope["margin_rectangles"](mx,my):
                cover[a:b,c:d]+=1
                assert 0<b-a<=32
            assert not cover[mx:mx+1920,my:my+1080].any()
            assert cover.max()<=1 and int(cover.sum())==cover.size-2073600



def test_b3_frame_archive_roundtrip_and_motion_reset(monkeypatch,tmp_path):
    from test_hybrid_cpu import b2_cpu_viewer
    v,clock=b2_cpu_viewer(monkeypatch,.04)
    v.hybrid_version=3;v.last_precise=None;v.margin_maps={}
    v.recorded=[];v.media=[];v.meta={"hybrid_version":3};v.log_dir=tmp_path;v.input_log=[]
    v.fast.precise_map=lambda *p:np.zeros((1920,1080),bool)
    class Chunk:
        def __init__(self,record,rect):
            self.rect=rect
        def poll(self):
            a,b,c,d=self.rect
            return np.ones((b-a,d-c),bool)
    v.margins=MarginScheduler(Chunk,[(0,1,0,1080),(1921,1922,0,1080)],1,0,v._margin_complete)
    def multi(display,candidates,transforms=None,version=2):
        assert version==3
        v.disp.bound_candidate_ids=[r["id"] for r,_ in candidates]
        return hybrid.candidate_plan3(display,[r for r,_ in candidates],transforms)[1:]
    v.disp.draw_candidates=multi
    def present():
        clock[0]+=.013
        return 0
    v.disp.present=present
    try:
        v.frame("initial",clock[0])
        v.frame("fast",clock[0])
        clock[0]+=.01
        v.frame("fast",clock[0])
        clock[0]+=.01
        v.frame("fast",clock[0])
        assert v.frames[-1]["candidates"][0]["part"]=="screen+margin"
        assert v.frames[-1]["candidates"][-1]["id"]=="precise_0"
        v.frame("precise",clock[0])
        assert v.shown_maps==[] and v.last_precise[0]["id"]=="precise_4"
        v.save()
        fresh,maps=vc.load_fresh(tmp_path)
        assert not any(vc.validate_present_records(v.frames,fresh,{"hybrid_version":3}).values())
        assert np.array_equal(maps[0],v.source_map) and maps.margins[0].shape==(1922,1080)
        for frame in v.frames[1:4]:
            expected=vc.displayed_map(frame,fresh,maps,{"hybrid_version":3})
            assert expected.shape==(1920,1080)
    finally:
        v.map_archive.close(clock[0])


def test_guard_retry_exhaustion_retains_final_contaminated_directory(monkeypatch,tmp_path):
    monkeypatch.setattr(cpu_guard,"wait_clean",lambda:{"waited_s":120,"reasons":[]})
    def launch(directory):
        directory.mkdir()
        (directory/"contamination.json").write_text(json.dumps({"verdict":{"contaminated":True}}))
        return directory
    with pytest.raises(cpu_guard.ContaminatedRun,match="contaminated: all 3 attempts"):
        cpu_guard.guarded_run(launch,tmp_path/"run")
    assert not (tmp_path/"run").exists()
    assert len(list(tmp_path.glob("run_contaminated_*")))==3
