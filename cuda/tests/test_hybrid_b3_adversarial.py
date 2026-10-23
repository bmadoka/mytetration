"""Adversarial CPU controls; never launches CUDA or a window."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_hybrid_b3_cpu import recording, sample, F
import cpu_guard
import viewer_check as vc
from margins import MarginScheduler


@pytest.mark.parametrize("damage", ["late", "missing", "gap", "overlap", "reversed", "nan",
                                    "pending", "screen", "download", "shape"])
def test_contradictory_margin_is_record_error(damage):
    frames,fresh,*_ = recording()
    r=fresh[0]
    if damage == "late":
        # Exact input: end 2.5, complete 4.0, decide 3.0, screen+margin.
        r["margin_chunks"][0]["t_complete"]=4.
    elif damage == "missing":
        r["margin_chunks"].pop()
    elif damage == "gap":
        r["margin_chunks"][0]["rect"][1]=1
    elif damage == "overlap":
        r["margin_chunks"].append(copy.deepcopy(r["margin_chunks"][0]))
    elif damage == "reversed":
        r["margin_chunks"][0]["t_launch"]=2.6
    elif damage == "nan":
        r["margin_chunks"][0]["t_complete"]=float("nan")
    elif damage == "pending":
        r["margin_chunks"][0]["t_complete"]=None
    elif damage == "screen":
        r["margin_chunks"][0]["rect"][1]=3
    elif damage == "download":
        r["margin_chunks"][0]["t_download_complete"]=4.
    elif damage == "shape":
        r["margin_shape"][0]+=1
    result=vc.validate_b3(frames,fresh)
    assert result["margin_records"] == [0]
    assert result["margin_ignored"] == [2]


def test_completion_partial_and_strict_usable_boundary():
    frames,fresh,*_=recording()
    r=fresh[0]
    assert vc.derived_margin_completion(r)==2.5
    frames[-1]["t_decide"]=2.5
    assert vc.b3_record_checks(frames,fresh)["margin_ignored"]==[2]
    r["margin_t_end"]=None
    r["margin_chunks"]=r["margin_chunks"][:1]
    assert vc.derived_margin_completion(r) is None
    assert vc.b3_record_checks(frames,fresh)["margin_ignored"]==[2]
    assert vc.b3_record_checks(frames,fresh)["margin_records"]==[]


def test_captured_nvidia_process_table_and_new_graphics_competitor(monkeypatch):
    captured=json.loads((F/"tests/data/hybrid_b3_ti_processes.json").read_text())
    table,devices,compute=captured["outputs"]
    rows=cpu_guard.parse_graphics_processes(table,1)
    assert rows and all(p["type"]=="C+G" for p in rows)
    assert 700 in {p["pid"] for p in rows} and not cpu_guard.parse_graphics_processes(table,9)
    assert {p["type"] for p in cpu_guard.parse_graphics_processes(table.replace("C+G", "G  "),1)}=={"G"}
    assert {p["type"] for p in cpu_guard.parse_graphics_processes(table.replace("C+G", "C  "),1)}=={"C"}
    linux="| Processes: |\n| 1 888 G /usr/bin/Xorg 125MiB |\n| 0 999 C python 32MiB |"
    assert cpu_guard.parse_graphics_processes(linux,1)==[{"pid":888,"name":"/usr/bin/Xorg","type":"G"}]
    seen=[]
    def run(command, **kwargs):
        seen.append(command)
        out=devices if "--query-gpu=" in command[1] else compute if "--query-compute" in command[1] else table
        return SimpleNamespace(stdout=out)
    monkeypatch.setattr(cpu_guard.subprocess,"run",run)
    monkeypatch.setattr(cpu_guard,"chosen_uuid",lambda:captured["chosen_uuid"])
    merged=cpu_guard.ti_processes()
    assert {p["pid"] for p in rows} <= {p["pid"] for p in merged}
    assert seen[-1][1]=="--id=00000000:33:00.0"
    baseline={p["pid"] for p in merged}
    assert not cpu_guard.verdict([sample(0,baseline)],baseline)["contaminated"]
    assert cpu_guard.verdict([sample(0,baseline|{987654})],baseline)["new_ti_pids"]==[987654]
    with pytest.raises(ValueError,match="process table missing"):
        cpu_guard.parse_graphics_processes("query failed",1)


def test_clean_window_continues_through_room_wait_and_actual_launch(monkeypatch):
    clock=[0.]
    tracked=[]
    def read(owned):
        return dict(sample(70 if 121 <= clock[0] <= 150 else 20,[700]),t=clock[0])
    sampler=cpu_guard.Sampler(source=SimpleNamespace(sample=read,track=tracked.append))
    sampler.started=sampler.clean_since=sampler.last_sample_t=0.
    sampler.wait_reasons=[]
    monkeypatch.setattr(cpu_guard.time,"perf_counter",lambda:clock[0])
    # Desktop/GPU/memory waits outlast the first 120 clean seconds. A competing
    # workload arrives during the wait and ends before desktop readiness.
    with sampler.changed:
        for second in range(1,152):
            clock[0]=float(second)
            sampler._record(read(sampler.owned))
    def wait(timeout):
        clock[0]+=1.
        sampler._record(read(sampler.owned))
    monkeypatch.setattr(sampler.changed,"wait",wait)
    launches=[]
    def start():
        launches.append(clock[0])
        return SimpleNamespace(pid=123456)
    sampler.launch(start)
    assert launches==[270.] and sampler.wait["waited_s"]==270.
    assert sampler.wait["clean_since"]==150. and len(sampler.wait["reasons"])==30
    assert tracked==[123456] and sampler.baseline=={700}
    trailing=[s for s in sampler.prelaunch if 150 < s["t"] <= 270]
    assert all(s["other_cpu"]<=50 for s in trailing)
    assert {s["t"] for s in trailing}==set(range(151,271))


def test_launch_room_wait_is_inside_sampler_and_boundary(monkeypatch,tmp_path):
    import launch
    events=[]
    class Sampler:
        def __init__(self):
            self.wait_reasons=[]
        def __enter__(self):
            events.append("sampling")
            return self
        def __exit__(self,*exc):
            events.append("sample end")
        def launch(self,factory):
            events.append("clean boundary")
            return factory()
        def write(self,path):
            events.append("report")
    class Child:
        pid=123456
        returncode=0
        def communicate(self,timeout):
            events.append("viewer wait")
            return "",""
        def terminate(self):
            pass
        def wait(self,timeout):
            pass
    def popen(command,**kwargs):
        events.append("viewer start" if command[0]!= "nvidia-smi" else "trace")
        return Child()
    def room(*args,**kwargs):
        events.append("room wait")
        return {}
    monkeypatch.setattr(launch.cpu_guard,"Sampler",Sampler)
    monkeypatch.setattr(launch,"wait_for_room",room)
    monkeypatch.setattr(launch.subprocess,"Popen",popen)
    launch.run("fake.json","timed",tmp_path/"run")
    assert events==["trace","sampling","room wait","clean boundary","viewer start","viewer wait","sample end","report"]


def test_window_replay_uses_same_guard_and_exposes_contamination(monkeypatch,tmp_path):
    import test_viewer as window_tests
    calls=[]
    monkeypatch.setattr(window_tests,"guarded_run",lambda launcher,directory:calls.append(directory) or launcher(directory))
    monkeypatch.setattr(window_tests.launch,"run",lambda *args,**kw:args[2])
    assert window_tests.replay(tmp_path,"D1","timed","none")==tmp_path/"viewer/D1_timed"
    assert calls==[tmp_path/"viewer/D1_timed"]
    directory=tmp_path/"record"
    directory.mkdir()
    report={"verdict":{"contaminated":True}}
    (directory/"contamination.json").write_text(json.dumps(report))
    meta=dict(hybrid_version=1,input_t={},skipped_same_params=0,fast_calls_playback=0,
              events_delivered=0,precision_calls_playback=[],pool_max_bytes=0,cuda_context_mib=0,
              gl_context_mib=0,foreground_is_viewer_after_show=False,mode="timed")
    monkeypatch.setattr(vc,"load",lambda d:(meta,[],{"start":[0.,0.,1.],"events":[]}))
    monkeypatch.setattr(vc,"load_fresh",lambda d:({},{}))
    monkeypatch.setattr(vc,"check_hybrid_records",lambda *args:{})
    assert vc.check_run(directory,recheck=False,precise_check=False)["contamination"]==report
    clean=dict(param_mismatch=[],started_before_input=[],sha_recheck_failed=[],calls_equal_counted=True,
               last_frame_is_precise_of_last_input=True,precise_checks=[],foreground_is_viewer_after_show=False,
               window_ok=True,present_failed=[],contamination=report)
    assert window_tests.violations(clean)==["contaminated"]
    from fault_judge import _is_intended
    assert not _is_intended(dict(when="call",exc_type="builtins.AssertionError",raised_in_test_body=True,
                                at_assert_statement=True,message_first_line="contaminated SIG:params"))


def test_busy_margin_limits_and_continuous_changed_inputs_complete_rings(monkeypatch):
    import margins
    clock=[0.]
    monkeypatch.setattr(margins.time,"perf_counter",lambda:clock[0])
    class Chunk:
        def __init__(self,r,rect):
            self.rect=rect
        def poll(self):
            a,b,c,d=self.rect
            return np.ones((b-a,d-c),bool)
    # Default 480x270 ring: 600 busy 8-row chunks. Continuous changed inputs
    # show a new map every 16 presents; retain only the latest 64 candidates.
    mx,my=480,270
    rects=[(0,mx,0,1620),(mx+1920,2880,0,1620),
           (mx,mx+1920,0,my),(mx,mx+1920,my+1080,1620)]
    scheduler=MarginScheduler(Chunk,rects,mx,my,mode="priority")
    records=[]
    screen=np.zeros((1920,1080),bool)
    for present in range(1300):
        if present % 16 == 0:
            ident=len(records)
            r={"id":ident,"params":(ident*.001,0.,1.)}
            records.append(r)
            scheduler.shown(r,screen,present)
            scheduler.retain({r["id"] for r in records[-64:]})
        clock[0]+=1
        scheduler.tick(True)
        assert len(scheduler.active)<=margins.MARGIN_INFLIGHT_BUSY==2
    completed=[r for r in records if r["margin_t_end"] is not None]
    assert len(completed)>=2
    ends=[r["margin_t_end"] for r in completed]
    assert all(b-a<=302 for a,b in zip(ends,ends[1:]))
    for r in records:
        for chunk in r["margin_chunks"]:
            assert chunk["rect"][1]-chunk["rect"][0]<=8
            assert chunk["mode"]=="priority" and chunk["screen_inflight"] is True
    for r in completed:
        assert len(r["margin_chunks"])==600
        assert vc.derived_margin_completion(r)==r["margin_t_end"]
    # Idle issues up to four <=32 rows; switching busy never adds to an existing
    # idle backlog. Previously issued GPU work cannot be cancelled/preempted.
    scheduler2=MarginScheduler(lambda r,rect:SimpleNamespace(poll=lambda:None),
                               [(0,128,0,1080)],128,0,mode="priority")
    r={"id":0}
    scheduler2.shown(r,screen,0)
    scheduler2.tick(False)
    assert len(scheduler2.active)==4 and all(c["rect"][1]-c["rect"][0]==32 for c in r["margin_chunks"])
    scheduler2.tick(True)
    assert len(r["margin_chunks"])==4


def test_gate_environment_retains_strict_host_gate(monkeypatch):
    monkeypatch.setenv("MYTET_MARGIN_MODE","gate")
    scheduler=MarginScheduler(lambda *args:pytest.fail("gate issued margin while busy"),[(0,1,0,1080)],1,0)
    scheduler.shown({"id":0},np.zeros((1920,1080),bool),0)
    scheduler.tick(True)
    assert scheduler.mode=="gate" and not scheduler.active


def test_sampler_short_run_boundary_and_failed_sample_never_clean():
    sampler=cpu_guard.Sampler(source=SimpleNamespace(sample=lambda owned:dict(sample(0,[700,999]),t=2.)))
    sampler.thread=SimpleNamespace(join=lambda:None)
    sampler.baseline={700}
    sampler.launch_t=1.5
    sampler.samples=[dict(sample(0,[700]),t=1.)]  # prelaunch rows are not runtime evidence
    sampler.__exit__()
    assert sampler.verdict["new_ti_pids"]==[999]
    def fail(owned):
        raise RuntimeError("query failed")
    failed=cpu_guard.Sampler(source=SimpleNamespace(sample=fail))
    failed.thread=SimpleNamespace(join=lambda:None)
    with pytest.raises(RuntimeError,match="contaminated"):
        failed.__exit__()
    assert failed.verdict["contaminated"] and failed.verdict["errors"]==["query failed"]


def test_honesty_window_also_rejects_all_contaminated_attempts(monkeypatch,tmp_path):
    import launch
    monkeypatch.setattr(vc,"load",lambda d:({"script":"unused","hybrid_version":1},[],{}))
    def run(script,mode,directory,**kwargs):
        assert mode=="honesty"
        directory.mkdir()
        (directory/"contamination.json").write_text(json.dumps({"verdict":{"contaminated":True}}))
        return directory
    monkeypatch.setattr(launch,"run",run)
    with pytest.raises(cpu_guard.ContaminatedRun,match="contaminated"):
        vc.check_replay(tmp_path,tmp_path/"honesty")
    assert len(list(tmp_path.glob("honesty_contaminated_*")))==3


def test_real_cpu_viewer_continuous_inputs_use_completed_margin(monkeypatch):
    import hybrid
    from test_hybrid_cpu import b2_cpu_viewer
    v,clock=b2_cpu_viewer(monkeypatch,.04)
    v.hybrid_version=3
    v.last_precise=None
    v.margin_maps={}
    class Chunk:
        def __init__(self,r,rect):
            self.rect=rect
        def poll(self):
            a,b,c,d=self.rect
            return np.ones((b-a,d-c),bool)
    v.margins=MarginScheduler(Chunk,[(0,4,0,1080),(1924,1928,0,1080)],4,0,v._margin_complete,mode="priority")
    class Job:
        def __init__(self,*p):
            self.t_start,self.precision=clock[0],"f64"
        def poll(self):
            return np.zeros((1920,1080),bool) if clock[0]-self.t_start >= .04 else None
    v.fast.FreshJob=Job
    def draw(display,candidates,transforms=None,version=3):
        v.disp.bound_candidate_ids=[r["id"] for r,_ in candidates]
        return hybrid.candidate_plan3(display,[r for r,_ in candidates],transforms)[1:]
    v.disp.draw_candidates=draw
    try:
        for i in range(16):
            clock[0]+=.02
            v.state=(i*.03,0.,50.)
            v.applied+=1
            v.frame("fast",clock[0])
        assert len(v.fresh)>=5
        assert sum(r.get("margin_t_end") is not None for r in v.fresh)>=3
        assert any(f["pixel_counts"]["margin"]>0 for f in v.frames)
        busy=[c for r in v.fresh for c in r.get("margin_chunks",[]) if c["screen_inflight"]]
        assert busy and all(c["mode"]=="priority" for c in busy)
    finally:
        v.map_archive.close(clock[0])


def test_concurrent_benchmark_discards_warmup_and_medians_without_gpu():
    import ast
    import hashlib
    tree=ast.parse((F/"bench/streams_margin.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="concurrent_screen_measurement")
    result=np.zeros((2,2),bool)
    alone=iter([1000.,1.,2.,3.])
    overlap=iter([2000.,4.,5.,6.])
    calls=[]
    def next_screen(p,previous_p,previous_map):
        calls.append(previous_p is not None)
        return result,next(overlap if previous_p is not None else alone),{"margin_chunks":[]}
    scope=dict(np=np,hashlib=hashlib,next_screen=next_screen,fast=SimpleNamespace(MX=480,MY=270),
               MARGIN_INFLIGHT_BUSY=2,MARGIN_ROWS_BUSY=8)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"bench_control","exec"),scope)
    report=scope[fn.name]((0.,0.,1.),(1.,0.,1.),result,hashlib.sha256(result.tobytes()).hexdigest())
    assert report["alone_median_ms"]==2. and report["with_margin_median_ms"]==5.
    assert report["samples"]==3 and report["warmup_dropped"]==1 and len(report["trials"])==4
    assert calls==[False,True,True,False,False,True,True,False]
