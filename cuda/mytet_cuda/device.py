"""Which NVIDIA GPU to use — decided from nvidia-smi alone, before CUDA is touched.

The GPU is named by its nvidia-smi index: command-line option `--gpu N`, or environment variable MYTET_GPU=N
(default 0). An index that is not a plain non-negative integer listed by nvidia-smi is refused with
GpuChoiceError before any CUDA library is loaded. mytet_cuda.gpu then makes only that GPU visible to CUDA
(bind(): CUDA_VISIBLE_DEVICES = its UUID) and checks the UUID again after CUDA starts.

The GPU list (index, UUID, name) is read from nvidia-smi once per process. Memory values (free / used MiB) are
never kept: memory() and query() ask nvidia-smi again on every call.

Optional machine rule: a file `local_gpu.json` next to the mytet_cuda/ folder (it is not part of the published
package). It only acts when nvidia-smi lists one of the GPUs it names: then the default (no --gpu, no MYTET_GPU)
is its "default" GPU, and the GPUs in its "never" list are refused for the default, MYTET_GPU and --gpu. A rule
file that cannot be read, or whose default GPU is not listed, stops the choice (no fallback to index 0).
A copy that also has the marker file `local_gpu.required` next to the mytet_cuda/ folder (also not published) is
made for that machine only: there a missing rule file, or a rule none of whose GPUs nvidia-smi lists, also stops the
choice. Without the marker (the published package) nothing changes: no rule file -> MYTET_GPU, else index 0.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

GPU_ENV = "MYTET_GPU"
MIN_FREE_MIB = 2048  # GPU work is not started when the chosen GPU has less free memory than this
LOCAL_FILE = Path(__file__).resolve().parents[1] / "local_gpu.json"
LOCAL_MARK = Path(__file__).resolve().parents[1] / "local_gpu.required"  # this copy must run under LOCAL_FILE

_gpus = None  # [{"index", "uuid", "name"}] of this process (one nvidia-smi query)
_rule = None  # parsed LOCAL_FILE of this process, or False when there is none


class GpuChoiceError(RuntimeError):
    pass


def _smi(fields):
    """Rows of `nvidia-smi --query-gpu=<fields>` (no CUDA call); GpuChoiceError when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        rows = [[s.strip() for s in line.split(",")] for line in out.strip().splitlines()]
        if not rows or any(len(r) != len(fields.split(",")) for r in rows):
            raise ValueError(f"unexpected output {out[:200]!r}")
        return rows
    except (OSError, subprocess.SubprocessError, ValueError) as e:  # no nvidia-smi, failed run, unexpected output
        raise GpuChoiceError(f"cannot read the GPU list with nvidia-smi ({type(e).__name__}: {e})") from None


def list_gpus():
    """[{"index", "uuid", "name"}] from nvidia-smi (no CUDA call), queried once per process."""
    global _gpus
    if _gpus is None:
        try:
            _gpus = [{"index": int(i), "uuid": u, "name": n} for i, u, n in _smi("index,uuid,name")]
        except ValueError as e:
            raise GpuChoiceError(f"cannot read the GPU list with nvidia-smi (ValueError: {e})") from None
    return [dict(g) for g in _gpus]


def memory(uuid):
    """{"free_mib", "used_mib"} of the GPU with this UUID now (fresh nvidia-smi query, never cached)."""
    for u, free, used in _smi("uuid,memory.free,memory.used"):
        if u == uuid:
            try:
                return {"free_mib": int(free), "used_mib": int(used)}
            except ValueError as e:
                raise GpuChoiceError(f"cannot read the GPU memory with nvidia-smi (ValueError: {e})") from None
    raise GpuChoiceError(f"GPU {uuid} is no longer listed by nvidia-smi")


def query(uuid):
    """Entry of the GPU with this UUID with its memory now: {"index", "uuid", "name", "free_mib", "used_mib"}."""
    for g in list_gpus():
        if g["uuid"] == uuid:
            return {**g, **memory(uuid)}
    raise GpuChoiceError(f"GPU {uuid} is not in the GPU list of this process")


def _exists(path):
    try:
        return path.exists()
    except OSError as e:
        raise GpuChoiceError(f"cannot check the local GPU rule file {path.name} ({type(e).__name__}: {e})") from None


def marked():
    """True when this copy carries LOCAL_MARK (it must run under its machine rule LOCAL_FILE)."""
    return _exists(LOCAL_MARK)


def local_rule():
    """The machine rule of LOCAL_FILE ({"default": uuid, "never": [uuid, ...]}), or None when there is no file.
    A copy with LOCAL_MARK but no LOCAL_FILE stops here (GpuChoiceError): no GPU is chosen."""
    global _rule
    if _rule is None:
        if not _exists(LOCAL_FILE):
            if marked():
                raise GpuChoiceError(f"{LOCAL_FILE.name} is missing but this copy has {LOCAL_MARK.name} "
                                     "(made for one machine only); no GPU chosen")
            _rule = False
        else:
            try:
                r = json.loads(LOCAL_FILE.read_text(encoding="utf-8"))
                ok = (isinstance(r, dict) and isinstance(r.get("default"), str) and isinstance(r.get("never"), list)
                      and all(isinstance(u, str) for u in r["never"]) and r["default"] not in r["never"])
                if not ok:
                    raise ValueError('needs {"default": "<uuid>", "never": ["<uuid>", ...]}')
            except (OSError, ValueError) as e:
                raise GpuChoiceError(f"cannot read the local GPU rule {LOCAL_FILE.name} ({type(e).__name__}: {e})") from None
            _rule = {"default": r["default"], "never": list(r["never"])}
    return _rule or None


def _here(gpus):
    """The machine rule when nvidia-smi lists one of its GPUs (this is the machine it was written for), else None.
    A copy with LOCAL_MARK has no «else»: its rule must apply here, otherwise GpuChoiceError (no GPU chosen)."""
    rule = local_rule()
    if rule and any(g["uuid"] == rule["default"] or g["uuid"] in rule["never"] for g in gpus):
        return rule
    if rule and marked():
        raise GpuChoiceError(f"nvidia-smi lists none of the GPUs of {LOCAL_FILE.name} and this copy has "
                             f"{LOCAL_MARK.name} (made for one machine only); no GPU chosen")
    return None


def machine_rule():
    """The machine rule of LOCAL_FILE when it applies to this machine (nvidia-smi lists one of its GPUs), else None."""
    return _here(list_gpus())


def _allowed(g, rule):
    if rule and g["uuid"] in rule["never"]:
        raise GpuChoiceError(f"GPU {g['index']} ({g['name']}) is excluded on this machine by {LOCAL_FILE.name}")
    return g


def choose(text=None):
    """The GPU named by `text` (default: $MYTET_GPU, else the machine rule's default, else "0"), as a dict of
    list_gpus(). An explicit `text` is a plain index lookup; the default / MYTET_GPU choice also obeys the rule."""
    explicit = text is not None
    if text is None:
        text = os.environ.get(GPU_ENV)
    if text is not None:
        s = str(text).strip()
        if not re.fullmatch(r"[0-9]{1,6}", s):  # refuses "-1", "a", "1.0", "", absurdly long numbers, before anything else
            raise GpuChoiceError(f"GPU index must be a non-negative integer (an nvidia-smi index), got {str(text)[:20]!r}")
    gpus = list_gpus()
    rule = _here(gpus)
    if text is None:
        if rule:
            for g in gpus:
                if g["uuid"] == rule["default"]:
                    return g
            raise GpuChoiceError(f"the default GPU of {LOCAL_FILE.name} is not listed by nvidia-smi")
        s = "0"
    for g in gpus:
        if g["index"] == int(s):
            return g if explicit else _allowed(g, rule)
    raise GpuChoiceError(f"no GPU with index {int(s)}; nvidia-smi lists {[g['index'] for g in gpus]}")


def bound():
    """The GPU this process's CUDA is already bound to (mytet_cuda.gpu imported), else None."""
    gpu = sys.modules.get("mytet_cuda.gpu")
    return getattr(gpu, "DEVICE", None)


def bind():
    """Choose this process's GPU (choose()) and make it the only GPU CUDA can see, here and in child processes
    (CUDA_VISIBLE_DEVICES = its UUID). Must run before CUDA is loaded; mytet_cuda.gpu calls it on import, and
    viewer/launch.py before it starts a viewer process. Refused when this process's CUDA is bound to another GPU."""
    g = choose()
    b = bound()
    if b is not None and b["uuid"] != g["uuid"]:
        raise GpuChoiceError(f"this process already uses GPU {b['index']} ({b['name']}); cannot switch to GPU {g['index']}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = g["uuid"]
    return g


def use(text):
    """Command-line option --gpu: check it, then make it the choice of this process and its children.
    Refused when CUDA in this process is already bound to another GPU (mytet_cuda.gpu was imported before)."""
    g = _allowed(choose(text), _here(list_gpus()))
    b = bound()
    if b is not None and b["uuid"] != g["uuid"]:
        raise GpuChoiceError(f"this process already uses GPU {b['index']} ({b['name']}); cannot switch to GPU {g['index']}")
    os.environ[GPU_ENV] = str(g["index"])
    return g
