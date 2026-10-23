"""Host-side yielding ring scheduler; no CUDA import, fakeable chunk factory."""
import os
import time

MARGIN_INFLIGHT_BUSY = 2
MARGIN_INFLIGHT_IDLE = 4
MARGIN_ROWS_BUSY = 8
MARGIN_ROWS_IDLE = 32

import numpy as np


class MarginScheduler:
    def __init__(self, factory, rectangles, mx, my, on_complete=None, *, mode=None):
        self.mode = mode or os.environ.get("MYTET_MARGIN_MODE", "priority")
        if self.mode not in ("priority", "gate"):
            raise ValueError("MYTET_MARGIN_MODE must be priority or gate")
        self.factory, self.rectangles = factory, rectangles
        self.mx, self.my = mx, my
        self.on_complete = on_complete
        self.tasks = {}
        self.active = []

    def shown(self, record, screen, seq):
        ident = record["id"]
        task = self.tasks.get(ident)
        if ((task is not None and not task.get("cancelled"))
                or record.get("margin_t_end") is not None or not (self.mx or self.my)):
            return
        attempt = {"status":"running", "margin_chunks":[], "margin_t_end":None}
        record.setdefault("margin_attempts", []).append(attempt)
        record.update(margin_shape=[1920+2*self.mx,1080+2*self.my],
                      margin_width=[self.mx,self.my], margin_t_end=None,
                      margin_chunks=attempt["margin_chunks"])
        self.tasks[ident] = {"record":record,"attempt":attempt,"screen":screen,"seq":seq,
                             "remaining":list(self.rectangles),"map":None,"inflight":0}

    def retain(self, ids):
        # Do not start uncomputed chunks for maps no longer candidates. In-flight
        # chunks keep their task alive until their observed completion is recorded.
        for ident in list(self.tasks):
            if ident not in ids:
                if self.tasks[ident]["attempt"]["status"] == "completed":
                    del self.tasks[ident]  # keep the completed record valid for earlier presents
                    continue
                self.tasks[ident]["remaining"].clear()
                self.tasks[ident]["cancelled"] = True
                self.tasks[ident]["attempt"]["status"] = "cancelled"
        self._discard()

    def _discard(self):
        for ident in list(self.tasks):
            task=self.tasks[ident]
            if task.get("cancelled") and not task["inflight"]:
                del self.tasks[ident]

    def tick(self, screen_inflight, *, launch=True):
        for task, chunk, row in list(self.active):
            data=chunk.poll()
            if data is None:
                continue
            row["t_complete"] = time.perf_counter()
            x0,x1,y0,y1=row["rect"]
            task["map"][x0:x1,y0:y1] = data
            task["inflight"] -= 1
            self.active = [item for item in self.active if item[1] is not chunk]
            if not task["remaining"] and not task["inflight"] and not task.get("cancelled"):
                task["map"].setflags(write=False)
                end = time.perf_counter()
                task["attempt"].update(status="completed", margin_t_end=end)
                task["record"]["margin_t_end"] = end
                if self.on_complete:
                    self.on_complete(task["record"],task["map"])
        self._discard()
        if not launch or (screen_inflight and self.mode == "gate"):
            return
        limit = MARGIN_INFLIGHT_BUSY if screen_inflight else MARGIN_INFLIGHT_IDLE
        rows = MARGIN_ROWS_BUSY if screen_inflight else MARGIN_ROWS_IDLE
        pending = [t for t in self.tasks.values() if not t.get("cancelled")
                   and t["attempt"]["status"] == "running"]
        # A started ring owns the scheduler across busy/idle transitions. Only
        # queued, unstarted rings compete newest-first. Eviction releases it.
        for task in sorted(pending,key=lambda t:(bool(t["attempt"]["margin_chunks"]),
                                                 t["seq"],t["record"]["id"]),reverse=True):
            while task["remaining"] and len(self.active)<limit:
                if task["map"] is None:
                    task["map"] = np.empty(task["record"]["margin_shape"],bool)
                    task["map"][self.mx:self.mx+1920,self.my:self.my+1080] = task["screen"]
                rect=task["remaining"].pop(0)
                a,b,c,d=rect
                if b-a > rows:
                    task["remaining"].insert(0,(a+rows,b,c,d))
                    rect=(a,a+rows,c,d)
                row={"rect":list(rect),"t_launch":time.perf_counter(),"t_complete":None,
                     "mode":self.mode,"screen_inflight":bool(screen_inflight)}
                task["attempt"]["margin_chunks"].append(row)
                chunk=self.factory(task["record"],rect)
                task["inflight"] += 1
                self.active.append((task,chunk,row))
            break  # finish this ring before starting another, even with free slots

    def completed(self, ident):
        task=self.tasks.get(ident)
        return task["map"] if task and task["record"].get("margin_t_end") is not None else None

    def drain_active(self):
        # Unlaunched work is optional; teardown waits only for already issued work.
        for task in self.tasks.values():
            if task["remaining"]:
                task["cancelled"] = True
                task["attempt"]["status"] = "cancelled"
                task["remaining"].clear()
        while self.active:
            self.tick(True, launch=False)
            if self.active:
                time.sleep(.001)
