"""CPU regressions: isolated attempts and per-loop margin progress."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from test_hybrid_b3_cpu import recording, times
from margins import MarginScheduler
import viewer_check as vc


@pytest.mark.parametrize("restart_pending", [False, True])
def test_pause_precise_resume_isolates_cancelled_attempt(monkeypatch, restart_pending):
    import margins
    clock=[2.]
    monkeypatch.setattr(margins.time,"perf_counter",lambda:clock[0])
    ready=[False]
    class Chunk:
        def __init__(self, record, rect):
            self.rect=rect
        def poll(self):
            a,b,c,d=self.rect
            return np.ones((b-a,d-c),bool) if ready[0] else None
    frames,fresh,*_=recording()
    r=fresh[0]
    for key in ("margin_chunks","margin_t_end"):
        r.pop(key)
    rects=[(0,64,0,1080),(1984,2048,0,1080)]
    completions=[]
    scheduler=MarginScheduler(Chunk,rects,64,0,lambda *args:completions.append(args),mode="priority")
    screen=np.zeros((1920,1080),bool)
    scheduler.shown(r,screen,1)
    scheduler.tick(True)  # two old chunks in flight
    old=r["margin_attempts"][0]
    assert len(old["margin_chunks"])==2
    clock[0]=2.1
    scheduler.retain(set())  # precise present cancels motion candidates
    assert old["status"]=="cancelled" and scheduler.completed(0) is None
    ready[0]=True
    if not restart_pending:
        scheduler.tick(True,launch=False)
        assert 0 not in scheduler.tasks
    clock[0]=2.2
    scheduler.shown(r,screen,3)  # resume same source before replacement finishes
    final=r["margin_attempts"][-1]
    assert final is not old and final["status"]=="running"
    scheduler.tick(False)
    assert r["margin_t_end"] is None and not completions
    clock[0]=2.4
    scheduler.tick(False)
    assert len(completions)==1 and len(final["margin_chunks"])==4
    assert old["status"]=="cancelled" and final["status"]=="completed"
    assert r["margin_chunks"] is final["margin_chunks"]
    assert vc.derived_margin_completion(r)==2.4
    # Represent the actual pause/precise/resume order in the independent checker.
    precise=copy.deepcopy(frames[0])
    precise.update(seq=2,kind="precise",present_kind="precise",t_decide=2.04,t_present=2.05)
    times(precise,2.06)
    frames.insert(2,precise)
    frames[-1]["seq"]=3
    assert not any(vc.validate_b3(frames,fresh).values())
    # A marked cancelled attempt is historical only, including partial tiling.
    old["margin_chunks"].append(copy.deepcopy(old["margin_chunks"][0]))
    assert not any(vc.validate_b3(frames,fresh).values())
    for damage in ("time","overlap","unmarked"):
        bad=copy.deepcopy(fresh)
        attempts=bad[0]["margin_attempts"]
        if damage=="time":
            attempts[-1]["margin_chunks"][0]["t_complete"]=4.
        elif damage=="overlap":
            attempts[-1]["margin_chunks"].append(copy.deepcopy(attempts[-1]["margin_chunks"][0]))
        else:
            attempts[0]["status"]="running"
        assert vc.validate_b3(frames,bad)["margin_records"]==[0]
        assert vc.validate_b3(frames,bad)["margin_ignored"]==[3]


def test_started_ring_survives_idle_switch_and_eviction_releases_newest(monkeypatch):
    import margins
    monkeypatch.setattr(margins.time,"perf_counter",lambda:1.)
    issued=[]
    class Chunk:
        def __init__(self,r,rect):
            self.rect=rect
            issued.append(r["id"])
        def poll(self):
            a,b,c,d=self.rect
            return np.ones((b-a,d-c),bool)
    scheduler=MarginScheduler(Chunk,[(0,128,0,1080),(2048,2176,0,1080)],128,0,mode="priority")
    screen=np.zeros((1920,1080),bool)
    a,b,c=[{"id":i} for i in range(3)]
    scheduler.shown(a,screen,0)
    scheduler.tick(True)
    scheduler.shown(b,screen,1)
    scheduler.shown(c,screen,2)
    scheduler.tick(False)
    assert issued==[0]*6  # idle does not preempt a started ring
    scheduler.retain({1,2})
    scheduler.tick(True)
    assert issued[-2:]==[2,2]  # newest unstarted ring wins after eviction
    assert a["margin_attempts"][0]["status"]=="cancelled"
    assert scheduler.completed(0) is None
    scheduler.drain_active()
    assert not scheduler.active
    assert c["margin_attempts"][0]["status"]=="cancelled"


def test_main_loop_polls_before_and_after_even_without_presents(monkeypatch):
    import app
    v=app.Viewer.__new__(app.Viewer)
    ticks=[]
    iterations=[0]
    def closed(win):
        iterations[0]+=1
        return iterations[0]>3
    v.__dict__.update(glfw=SimpleNamespace(window_should_close=closed,poll_events=lambda:None),
                      disp=SimpleNamespace(win=None),_poll_fresh=lambda:None,
                      hybrid_version=3,margins=SimpleNamespace(tick=lambda busy,**kw:ticks.append((busy,kw))),
                      job=object(),display=(0.,0.,1.),state=(0.,0.,1.),hybrid_mode="warped",
                      last_kind="initial",applied=0,drawn=0)
    monkeypatch.setattr(app.time,"sleep",lambda _:None)
    v.frame=lambda *args:pytest.fail("idle loop drew a frame")
    v._loop(None,None,None,0.,0)
    assert ticks==[(True,{"launch":False}),(True,{"launch":True})]*3  # launch only in warped mode
