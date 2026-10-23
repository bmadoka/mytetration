"""All-core Windows CPU and chosen-GPU contamination guard; no psutil."""
import csv
import ctypes
from ctypes import wintypes
import json
import os
import re
from pathlib import Path
import subprocess
import threading
import time

def chosen_uuid():
    """UUID of the GPU chosen with --gpu / MYTET_GPU (nvidia-smi only, no CUDA)."""
    from mytet_cuda import device
    return device.choose()["uuid"]


def __getattr__(name):  # TI_UUID = the chosen GPU's UUID, read when used (after --gpu has been applied)
    if name == "TI_UUID":
        return chosen_uuid()
    raise AttributeError(name)



def parse_ti_processes(text):
    uuid = chosen_uuid()
    rows = []
    for row in csv.reader(text.splitlines()):
        if len(row) >= 3 and row[0].strip() == uuid:
            rows.append({"pid": int(row[1].strip()), "name": ",".join(row[2:]).strip()})
    return rows


def parse_graphics_processes(text, ti_index):
    """Default nvidia-smi process table (WDDM and Linux G/C/C+G rows)."""
    rows = []
    in_table = False
    for line in text.splitlines():
        if "Processes:" in line:
            in_table = True
        if not in_table:
            continue
        # GPU, optional GI/CI, PID, type, executable, memory (N/A on WDDM).
        match = re.match(r"^\s*\|\s*(\d+)\s+(?:(?:N/A|\d+)\s+(?:N/A|\d+)\s+)?"
                         r"(\d+)\s+(C\+G|G|C)\s+(.+?)\s+(?:N/A|\d+MiB)\s*\|\s*$", line)
        if not match and re.match(r"^\s*\|\s*\d+\s+", line):
            raise ValueError("unrecognized nvidia-smi process row: "+line)
        if match and int(match[1]) == int(ti_index):
            rows.append({"pid":int(match[2]), "name":match[4].strip(), "type":match[3]})
    if not in_table:
        raise ValueError("nvidia-smi process table missing")
    return rows


def ti_processes():
    def query(*args):
        return subprocess.run(["nvidia-smi", *args],capture_output=True,text=True,check=True,timeout=10).stdout
    devices = list(csv.reader(query("--query-gpu=uuid,index,pci.bus_id", "--format=csv,noheader").splitlines()))
    ti = next((row for row in devices if row[0].strip() == chosen_uuid()), None)
    if ti is None:
        raise RuntimeError("chosen GPU UUID/index/bus not found")
    compute = parse_ti_processes(query("--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"))
    graphics = parse_graphics_processes(query("--id="+ti[2].strip()), ti[1].strip())
    return list({p["pid"]:p for p in compute + graphics}.values())


def _ticks(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


def process_parents():
    """pid -> parent pid of every live process (Toolhelp snapshot)."""
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    if snap in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        parents = {}
        ok = k.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = k.Process32NextW(snap, ctypes.byref(entry))
        return parents
    finally:
        k.CloseHandle(snap)


def descendants(roots, parents):
    """roots plus every live process whose parent chain reaches a root (e.g. the venv launcher's real python)."""
    out = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in out and pid not in out:
                out.add(pid)
                changed = True
    return out


class WindowsCounters:
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
        self.kernel.GetSystemTimes.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.last = None
        self.process_ticks = {}
        self.handles = {}

    def track(self,pid,strict=True):
        if pid not in self.handles:
            handle=self.kernel.OpenProcess(0x1000,False,int(pid))
            if not handle:
                if not strict:  # a descendant that already exited between snapshot and open
                    return False
                raise ctypes.WinError(ctypes.get_last_error())
            self.handles[pid]=handle
            if self.last is not None:
                self.process_ticks[pid]=0  # child born after the system baseline

    def close(self):
        for handle in self.handles.values():
            self.kernel.CloseHandle(handle)
        self.handles.clear()

    def sample(self, owned, processes=True):
        # Query graphics/compute first: the CPU snapshot and timestamp must be
        # immediately before Popen at the launch boundary, not before smi I/O.
        processes=ti_processes() if processes else []
        idle,kernel,user = (wintypes.FILETIME() for _ in range(3))
        if not self.kernel.GetSystemTimes(ctypes.byref(idle),ctypes.byref(kernel),ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        total,busy = _ticks(kernel)+_ticks(user),_ticks(kernel)+_ticks(user)-_ticks(idle)
        own_delta = 0
        if isinstance(owned, set):
            # Descendants of owned processes are ours (the venv python.exe launcher starts the real interpreter as a
            # child; that child is the viewer on the chosen GPU).
            for pid in descendants(owned, process_parents()) - owned:
                if self.track(pid, strict=False) is not False:
                    owned.add(pid)
        for pid in tuple(owned):
            self.track(pid)
            handle=self.handles[pid]  # held through process exit, preserving final CPU ticks
            create,exit_,pk,pu = (wintypes.FILETIME() for _ in range(4))
            if not self.kernel.GetProcessTimes(handle,*[ctypes.byref(v) for v in (create,exit_,pk,pu)]):
                raise ctypes.WinError(ctypes.get_last_error())
            ticks=_ticks(pk)+_ticks(pu)
            previous=self.process_ticks.get(pid)
            if previous is not None:
                own_delta += max(0,ticks-previous)
            self.process_ticks[pid]=ticks
        percent=0. if self.last is None else 100*max(0,busy-self.last[1]-own_delta)/max(1,total-self.last[0])
        self.last=(total,busy)
        return {"t":time.perf_counter(),"other_cpu":percent,"ti_processes":processes}


def verdict(samples, baseline=(), owned=(), limit=50.):
    baseline,owned=set(baseline),set(owned)
    high=sum(s["other_cpu"]>limit for s in samples)
    # A sampling gap hides seconds; count every missing second as a high second.
    times=sorted(s["t"] for s in samples if "t" in s)
    missing=sum(max(0,int(-(-(b-a)//1))-1) for a,b in zip(times,times[1:]) if b-a>1.5)
    high+=missing
    newcomers=sorted({p["pid"] for s in samples for p in s["ti_processes"]
                      if p["pid"] not in baseline | owned})
    fraction=high/(len(samples)+missing) if samples else 0.
    return {"contaminated":fraction>.05 or bool(newcomers),"samples":len(samples),"missing_seconds":missing,
            "high_cpu_seconds":high,"high_cpu_fraction":fraction,"new_ti_pids":newcomers,
            "limit":limit,"baseline_ti_pids":sorted(baseline),"owned_pids":sorted(owned)}


def wait_clean(window=120, limit=50., *, owned=None, source=None, sleep=time.sleep, now=time.perf_counter):
    owned=set(owned or [os.getpid()])
    source=source or WindowsCounters()
    start=clean=now()
    reasons=[]
    source.sample(owned)  # establish deltas, not a measured second
    while now()-clean < window:
        sleep(min(1.,window-(now()-clean)))
        row=source.sample(owned)
        if row["other_cpu"]>limit:
            reasons.append({"t":now(),"other_cpu":row["other_cpu"],"reason":"other CPU above limit"})
            clean=now()
    if hasattr(source,"close"):
        source.close()
    return {"waited_s":now()-start,"window_s":window,"limit":limit,"reasons":reasons}


class Sampler:
    def __init__(self, owned=None, *, source=None, baseline=None, interval=1.):
        self.owned=set(owned or [os.getpid()])
        self.source=source or WindowsCounters()
        self.baseline=None if baseline is None else set(baseline)
        self.interval=interval
        self.samples=[]
        self.errors=[]
        self.stop=threading.Event()
        self.lock=threading.RLock()
        self.changed=threading.Condition(self.lock)
        self.prelaunch=[]
        self.launch_t=None
        self.wait=None

    def add_owned(self,pid):
        with self.lock:
            if hasattr(self.source,"track"):
                self.source.track(pid)
            self.owned.add(pid)

    def _record(self, row):
        if self.launch_t is None and row["t"]-self.last_sample_t > self.interval*1.5:
            self.clean_since=row["t"]
            self.wait_reasons.append({"t":row["t"],"reason":"sampling gap before launch"})
        self.last_sample_t=row["t"]
        self.samples.append(row)
        if self.launch_t is None:
            self.prelaunch.append(row)
            if row["other_cpu"] > 50.:
                self.clean_since=row["t"]
                self.wait_reasons.append({"t":row["t"], "reason":"other CPU above limit",
                                          "other_cpu":row["other_cpu"]})
        self.changed.notify_all()

    def launch(self, factory, window=120.):
        """Keep sampling through room waits; check again at actual Popen boundary."""
        with self.changed:
            while True:
                if self.errors:
                    raise RuntimeError("contaminated: prelaunch sampler failed") from self.errors[0]
                now=time.perf_counter()
                monitor=getattr(self,"monitor",None)
                if monitor is not None:
                    # Start condition = the preceding `window` seconds are fully sampled and at most
                    # 5 % of those seconds have other CPU > 50 % (the same tolerance as the in-run contamination rule).
                    ok,info=monitor.start_window(window)
                    if not ok:
                        if info != getattr(self,"_last_start_info",None) and len(self.wait_reasons) < 2000:
                            self.wait_reasons.append({"t":now,"reason":"start window not clean","info":info})
                        self._last_start_info=info
                        self.changed.wait(timeout=1.)
                        continue
                    self.clean_since=min(self.clean_since,now-window)
                if now-self.clean_since >= window:
                    # Refresh the partial last second immediately before Popen.
                    # This closes the gap after desktop/GPU/memory waits.
                    try:
                        row=self.source.sample(self.owned)
                    except BaseException as error:
                        self.errors.append(error)
                        raise RuntimeError("contaminated: launch-boundary sample failed") from error
                    self._record(row)
                    if row["t"]-self.clean_since >= window:
                        self.baseline={p["pid"] for p in row["ti_processes"]}
                        child=factory()
                        self.launch_t=time.perf_counter()
                        self.add_owned(child.pid)
                        self.wait={"waited_s":self.launch_t-self.started,"window_s":window,"limit":50.,
                                   "clean_since":self.clean_since,"launch_t":self.launch_t,
                                   "reasons":list(self.wait_reasons)}
                        return child
                self.changed.wait(timeout=min(self.interval, max(.001,window-(now-self.clean_since))))

    def __enter__(self):
        try:
            initial=self.source.sample(self.owned)
        except BaseException as error:
            self.errors.append(error)
            raise
        self.started=self.clean_since=self.last_sample_t=initial["t"]
        self.wait_reasons=[]
        self.monitor=process_monitor() if isinstance(self.source, WindowsCounters) else None
        if isinstance(self.source, WindowsCounters):
            # The 2-minute clean start is judged from a process-wide 1 s CPU monitor that keeps
            # running between launches, so back-to-back launches do not each wait 120 s from scratch.
            since = process_monitor().clean_since()
            if since is not None and since < self.clean_since:
                self.clean_since = since
                self.wait_reasons.append({"t": initial["t"], "reason": "clean since (process monitor)", "clean_since": since})
        if self.baseline is None:
            self.baseline={p["pid"] for p in initial["ti_processes"]}
        self.thread=threading.Thread(target=self._loop,name="cpu-guard",daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        try:
            deadline=time.perf_counter()+self.interval
            while not self.stop.wait(max(0.,deadline-time.perf_counter())):
                with self.changed:
                    self._record(self.source.sample(self.owned))
                deadline+=self.interval
        except BaseException as error:
            with self.changed:
                self.errors.append(error)
                self.changed.notify_all()

    def __exit__(self,*exc):
        self.stop.set()
        try:  # Bounded; a stalled monitor = contaminated, not a hang
            self.thread.join(timeout=30)
        except TypeError:  # test doubles without a timeout parameter
            self.thread.join()
        if getattr(self.thread, "is_alive", lambda: False)():
            self.errors.append(RuntimeError("cpu-guard sampler thread stalled"))
        # A short correctness smoke may have zero whole seconds. Still catch a
        # newly appeared GPU process at the boundary; do not claim timing from it.
        try:
            self.samples.append(self.source.sample(self.owned))
        except BaseException as error:
            self.errors.append(error)
        finally:
            if hasattr(self.source,"close"):
                self.source.close()
        if self.errors:
            raise RuntimeError("contaminated: sampler failed") from self.errors[0]

    @property
    def verdict(self):
        runtime = self.samples if self.launch_t is None else [s for s in self.samples if s["t"] >= self.launch_t]
        result=verdict(runtime,self.baseline or (),self.owned)
        if self.errors:
            result.update(contaminated=True,errors=[str(e) for e in self.errors])
        return result

    def write(self,path,wait=None):
        Path(path).write_text(json.dumps({"wait":self.wait if wait is None else wait,
            "prelaunch_samples":self.prelaunch,"samples":self.samples,"verdict":self.verdict},indent=1),encoding="utf-8")


class ContaminatedRun(RuntimeError):
    pass


def guarded_run(launcher, directory, *, attempts=3):
    """Shared by tests/measure/video; launcher enforces clean actual start."""
    if attempts < 1 or attempts > 3:
        raise ValueError("guard requires one to three attempts")
    directory=Path(directory)
    previous=[]
    for attempt in range(1,attempts+1):
        result=Path(launcher(directory))
        report=json.loads((result/"contamination.json").read_text(encoding="utf-8"))
        report["attempt"]=attempt
        report["previous_attempts"]=list(previous)
        (result/"contamination.json").write_text(json.dumps(report,indent=1),encoding="utf-8")
        if report["verdict"]["contaminated"] is False:
            return result
        renamed=directory.with_name(directory.name+f"_contaminated_{attempt}")
        if renamed.exists():
            raise FileExistsError(renamed)
        directory.rename(renamed)
        previous.append({"directory":str(renamed),"wait":report.get("wait"),"verdict":report["verdict"]})
    raise ContaminatedRun(f"contaminated: all {attempts} attempts retained: {previous}")


class _ProcessMonitor:
    """Process-wide 1 s CPU monitor (other CPU = all cores minus this process and its descendants)."""

    def __init__(self, limit=50.):
        self.counters = WindowsCounters()
        self.owned = {os.getpid()}
        self.limit = limit
        self.lock = threading.Lock()
        self._since = None
        self._last = None
        self.history = []  # (t, other_cpu) of the last ~10 minutes
        self.thread = threading.Thread(target=self._loop, name="cpu-guard-monitor", daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            try:
                row = self.counters.sample(self.owned, processes=False)
            except Exception:
                with self.lock:
                    self._since = self._last = None
                time.sleep(1.)
                continue
            with self.lock:
                t = row["t"]
                if self._last is None:
                    self._since = None  # first sample only sets the delta baseline
                elif t - self._last > 1.5 or row["other_cpu"] > self.limit:
                    self._since = t
                elif self._since is None:
                    self._since = self._last
                if self._last is not None:
                    self.history.append((t, row["other_cpu"]))
                    self.history = self.history[-600:]
                self._last = t
            time.sleep(max(0., 1. - (time.perf_counter() - t)))

    def start_window(self, window=120., limit=50., tolerance=.05):
        """(ok, info): the last `window` s are covered by 1 s samples without gaps and <= tolerance of them exceed limit."""
        with self.lock:
            now = time.perf_counter()
            rows = [(t, c) for t, c in self.history if t >= now - window]
            times = [t for t, _ in rows]
            covered = (bool(self.history) and self.history[0][0] <= now - window + 1.5 and self._last is not None
                       and now - self._last <= 1.5 and all(b - a <= 1.5 for a, b in zip(times, times[1:])))
            high = sum(c > limit for _, c in rows)
            ok = covered and high <= tolerance * window
            return ok, {"covered": covered, "samples": len(rows), "high_seconds": high}

    def clean_since(self):
        with self.lock:
            if self._last is None or time.perf_counter() - self._last > 1.5:
                return None
            return self._since


_MONITOR = None
_MONITOR_LOCK = threading.Lock()


def process_monitor():
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is None:
            _MONITOR = _ProcessMonitor()
        return _MONITOR
