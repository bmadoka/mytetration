"""Windowless B3 benchmark. --smoke runs ONE correctness state, no timing table."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

F=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(F/"viewer"),str(F/"scripts"),str(F)]
import fast
from margins import MarginScheduler, MARGIN_INFLIGHT_BUSY, MARGIN_ROWS_BUSY
from cpu_guard import Sampler, wait_clean


def states(records=None):
    records=Path(records) if records else F/"logs/hybrid_b2/measure_timing"
    result=[]
    for scene in ("DZa","DZb"):
        for kind in ("scroll","manip"):
            path=records/f"{scene}_{kind}_run2"/"presents.jsonl"
            if not path.is_file():
                raise SystemExit(f"needs recorded hybrid runs (presents.jsonl of DZa/DZb scroll/manip run 2, from viewer/measure.py); not found: {path}")
            rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for i in np.linspace(0,len(rows)-1,5,dtype=int):
                row=rows[i]
                result.append({"scene":scene,"script":kind,"seq":row["seq"],"params":row["display_params"]})
    return result


def screen(p,streams):
    previous=fast.STREAMS
    fast.STREAMS=streams
    try:
        job=fast.FreshJob(*p)
        job.event.synchronize()
        return job.poll()
    finally:
        fast.STREAMS=previous


def ring(p,m,mx,my):
    if not (mx or my):
        return {"pixels":0,"chunks":0}
    record={"id":0,"params":p,"precision":fast.choose_precision(*p)}
    scheduler=MarginScheduler(lambda r,rect:fast.MarginChunk(p,r["precision"],rect,mx,my),
                              fast.margin_rectangles(mx,my),mx,my)
    scheduler.shown(record,m,0)
    while record["margin_t_end"] is None:
        scheduler.tick(False)
        time.sleep(.0001)
    extended=scheduler.completed(0)
    assert np.array_equal(extended[mx:mx+1920,my:my+1080],m)
    return {"pixels":extended.size-m.size,"chunks":len(record["margin_chunks"]),
            "chunk_records":record["margin_chunks"],"margin_t_end":record["margin_t_end"]}


def next_screen(p, previous_p=None, previous_map=None):
    """Four-stream screen wall time, with optional previous-map margin traffic.

    Includes host submission, polling and async download as in the viewer. The
    scheduler is primed with up to two busy 8-row chunks before screen submission;
    subsequent ticks retain the same busy cap as the viewer. Drain is outside
    the screen timer.
    """
    scheduler = None
    record = None
    if previous_map is not None:
        record = {"id":0,"params":previous_p,"precision":fast.choose_precision(*previous_p)}
        scheduler = MarginScheduler(lambda r,rect:fast.MarginChunk(r["params"],r["precision"],rect),
                                    fast.margin_rectangles(),fast.MX,fast.MY,mode="priority")
        scheduler.shown(record,previous_map,0)
        scheduler.tick(True)
    streams = fast.STREAMS
    fast.STREAMS = 4
    try:
        started = time.perf_counter()
        job = fast.FreshJob(*p)
        result = job.poll()
        while result is None:
            if scheduler is not None:
                scheduler.tick(True)
            time.sleep(.0001)
            result = job.poll()
        elapsed = 1000*(time.perf_counter()-started)
    finally:
        fast.STREAMS = streams
        if scheduler is not None:
            scheduler.drain_active()
    return result, elapsed, record


def concurrent_screen_measurement(p, previous_p, previous_map, reference_sha):
    alone, concurrent, trials = [], [], []
    for rnd in range(4):  # first paired trial is warm-up, then median of three
        trial = {"warmup":rnd == 0}
        # Alternate order to avoid making all concurrent samples the later run.
        order = (False,True) if rnd % 2 == 0 else (True,False)
        for overlap in order:
            m, elapsed, record = next_screen(p, previous_p if overlap else None,
                                             previous_map if overlap else None)
            if hashlib.sha256(m.tobytes()).hexdigest() != reference_sha:
                raise AssertionError("concurrent next-screen SHA mismatch")
            key = "with_margin_ms" if overlap else "alone_ms"
            trial[key] = elapsed
            if overlap:
                trial["margin_chunks"] = record["margin_chunks"]
            if rnd:
                (concurrent if overlap else alone).append(elapsed)
        trials.append(trial)
    a,b = float(np.median(alone)),float(np.median(concurrent))
    return {"measurement":"screen job (4 streams) alone vs previous-map margin scheduler",
            "streams":4,"margin_mode":"priority","previous_params":previous_p,
            "margin_inflight_busy":MARGIN_INFLIGHT_BUSY,"margin_rows_busy":MARGIN_ROWS_BUSY,
            "margin_width":[fast.MX,fast.MY],"warmup_dropped":1,"samples":3,
            "alone_median_ms":a,"with_margin_median_ms":b,
            "difference_ms":b-a,"ratio":b/a,"sha_equal":True,"trials":trials}


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--records",help="folder with the recorded hybrid runs (default: logs/hybrid_b2/measure_timing)")
    ap.add_argument("--out",default=str(F/"bench/hybrid_b3/streams_margin.json"))
    args=ap.parse_args(argv)
    picks=states(args.records)[:1] if args.smoke else states(args.records)
    output={"smoke":args.smoke,"states":picks,"device":fast.gpu.device_info(),"rows":32,"attempts":[]}
    if fast.ROWS_PER_LAUNCH!=32:
        raise ValueError("benchmark requires unchanged 32-row chunks")
    attempts=1 if args.smoke else 3
    for attempt in range(attempts):
        wait=None if args.smoke else wait_clean()
        rows=[]
        with Sampler() as sampler:
            for state_index,state in enumerate(picks):
                p=state["params"]
                reference=fast.fast_map(*p)
                sha=hashlib.sha256(reference.tobytes()).hexdigest()
                for streams in (1,2,4):
                    values=[]
                    for rnd in range(1 if args.smoke else 4):
                        t=time.perf_counter()
                        m=screen(p,streams)
                        elapsed=(time.perf_counter()-t)*1000
                        actual=hashlib.sha256(m.tobytes()).hexdigest()
                        if actual!=sha:
                            raise AssertionError(f"screen SHA mismatch {state} streams={streams}")
                        if rnd>0:
                            values.append(elapsed)
                    rows.append(dict(state=state,streams=streams,sha_equal=True,sha256=sha,
                                     median_ms=float(np.median(values)) if values else None))
                if not args.smoke:
                    previous_p=picks[(state_index-1) % len(picks)]["params"]
                    previous_map=screen(previous_p,4)
                    rows.append(dict(state=state,concurrent_screen=concurrent_screen_measurement(
                        p,previous_p,previous_map,sha)))
                for mx,my in ((0,0),(240,135),(480,270),(960,540)):
                    values=[]
                    for rnd in range(1 if args.smoke else 4):
                        t=time.perf_counter()
                        detail=ring(p,reference,mx,my)
                        elapsed=(time.perf_counter()-t)*1000
                        if rnd>0:
                            values.append(elapsed)
                    rows.append(dict(state=state,margin=[mx,my],ring=detail,
                                     median_ms=float(np.median(values)) if values else None))
        output["attempts"].append({"wait":wait,"rows":rows,"samples":sampler.samples,"verdict":sampler.verdict})
        if args.smoke or not sampler.verdict["contaminated"]:
            break
    output["timing_measured"]=not args.smoke
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=1),encoding="utf-8")
    print(json.dumps({"output":str(path),"states":len(picks),"smoke":args.smoke,
                      "contamination":sampler.verdict}))


if __name__=="__main__":
    main()
