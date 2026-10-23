"""Launch one viewer process for a script, after the desktop/GPU/memory checks.

Before every launch: user idle >= 5 min (GetLastInputInfo), no full-screen app in front, the chosen GPU
(--gpu / MYTET_GPU, mytet_cuda.device) free >= 2,048 MiB,
FreeVirtualMemory >= 8,388,608 KB. If not, wait and re-check every 30 s; give up after `max_wait` seconds.
A 0.2 s nvidia-smi trace of that GPU runs alongside (reference only).
"""
import datetime as dt
import json
import io
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(F))
import desktop  # noqa: E402
sys.path.insert(0,str(F / "scripts"))
import cpu_guard  # noqa: E402
from mytet_cuda import device  # noqa: E402


ENV_NAMES_IGNORE_CASE = os.name == "nt"  # Windows: cuda_visible_devices and CUDA_VISIBLE_DEVICES are one variable


class NotNow(RuntimeError):
    pass


def viewer_env(env, chosen):
    """Environment of the viewer process: this process's environment + `env`. `env` must not choose another GPU than
    `chosen` (MYTET_GPU / CUDA_VISIBLE_DEVICES, ValueError). On Windows its names are upper-cased before the check
    (names ignore case there), and a name given twice in different case is refused."""
    extra = {}
    for name, value in (env or {}).items():
        key = name.upper() if ENV_NAMES_IGNORE_CASE else name
        if key in extra:
            raise ValueError(f"env gives {key!r} twice (names that differ only in case are one variable on Windows)")
        extra[key] = value
    if device.GPU_ENV in extra and str(extra[device.GPU_ENV]).strip() != str(chosen["index"]):
        raise ValueError(f"env must not choose another GPU than this process ({device.GPU_ENV}); use --gpu / {device.GPU_ENV}")
    if extra.get("CUDA_VISIBLE_DEVICES", chosen["uuid"]) != chosen["uuid"]:
        raise ValueError("env must not set another CUDA_VISIBLE_DEVICES than this process's GPU; use --gpu / "
                         f"{device.GPU_ENV}")
    child_env = dict(os.environ, MPLCONFIGDIR=str(F / ".mplconfig"))
    child_env.update(extra)
    return child_env


def gpu_free_mib():
    return device.memory(device.choose()["uuid"])["free_mib"]


def wait_for_room(max_wait=3600, log=None):
    t_end = time.time() + max_wait
    while True:
        ok, why = desktop.may_open_window()
        free = gpu_free_mib()
        mem = desktop.free_virtual_kb()
        if ok and free >= 2048 and mem >= 8_388_608:
            return {"idle_check": why, "gpu_free_mib": free, "free_virtual_kb": mem}
        msg = f"{dt.datetime.now():%H:%M:%S} waiting: {why}; GPU free {free} MiB; FreeVirtualMemory {mem} KB"
        if log:
            log.write(msg + "\n")
        print(msg, file=sys.stderr, flush=True)
        if time.time() > t_end:
            raise NotNow(msg)
        time.sleep(30)


def run(script, mode, log_dir, readback="none", max_wait=3600, timeout=300, replay=None, env=None):
    chosen = device.bind()  # this process's GPU (--gpu / MYTET_GPU / machine rule); the viewer inherits its CUDA mask
    child_env = viewer_env(env, chosen)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    trace = open(log_dir / "gpu_trace.csv", "w")
    tr = subprocess.Popen(["nvidia-smi", f"--id={chosen['uuid']}", "--query-gpu=timestamp,memory.used,utilization.gpu",
                           "--format=csv,noheader", "-lms", "200"], stdout=trace, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        command = [sys.executable, "-m", "viewer", "--script", str(script), "--mode", mode, "--readback", readback,
                   "--log", str(log_dir)]
        if replay is not None:
            command += ["--replay", str(replay)]
        sampler=cpu_guard.Sampler()
        try:
            with sampler:
                room_log=io.StringIO()
                pre = wait_for_room(max_wait, log=room_log)
                sampler.wait_reasons.append({"t":time.perf_counter(),"reason":"desktop/GPU/memory checks",
                                             "details":room_log.getvalue(),"pre_checks":pre})
                child=sampler.launch(lambda: subprocess.Popen(command,cwd=str(F),env=child_env,
                                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
                try:
                    stdout,stderr=child.communicate(timeout=timeout)
                except BaseException:
                    child.kill()
                    child.communicate()
                    raise
                cp=subprocess.CompletedProcess(command,child.returncode,stdout,stderr)
        finally:
            if sampler is not None:
                sampler.write(log_dir / "contamination.json")
    finally:
        tr.terminate()
        tr.wait(10)
        trace.close()
    (log_dir / "launch.json").write_text(json.dumps({"pre_checks": pre, "returncode": cp.returncode,
                                                     "wall_s": time.time() - t0, "stderr_tail": cp.stderr[-3000:]},
                                                    indent=1), encoding="utf-8")
    if cp.returncode != 0:
        raise RuntimeError(f"viewer exit {cp.returncode}: {cp.stderr[-2000:]}")
    return log_dir
