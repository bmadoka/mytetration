"""Perturb evaluation I/O and mandatory memory/device/measurement guards."""
import json
import os
from pathlib import Path
import subprocess
import sys

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
def chosen_uuid():
    """UUID of the GPU chosen with --gpu / MYTET_GPU (nvidia-smi only, no CUDA)."""
    from mytet_cuda import device
    return device.choose()["uuid"]


def __getattr__(name):  # TI_UUID = the chosen GPU's UUID, read when used (after --gpu has been applied)
    if name == "TI_UUID":
        return chosen_uuid()
    raise AttributeError(name)

DEPTHS = (1e-15, 1e-20, 1e-30, 1e-40)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def finite_json(value):
    """Represent unbounded orbit log diagnostics explicitly in portable JSON."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [finite_json(v) for v in value]
    return value


def require_memory():
    from viewer.desktop import free_virtual_kb
    free = free_virtual_kb()
    if free < 8388608:
        raise RuntimeError(f'FreeVirtualMemory {free} KB < 8388608 KB')
    return free


def audit_gpu(stage, directory, pid=None, require_present=True):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name',
                             '--format=csv,noheader'], capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        cells = [s.strip() for s in line.split(',', 2)]
        if len(cells) == 3:
            rows.append({'pid': int(cells[0]), 'gpu_uuid': cells[1], 'process_name': cells[2]})
    pid = os.getpid() if pid is None else pid
    own = [r for r in rows if r['pid'] == pid]
    uuid = chosen_uuid()
    record = {'stage': stage, 'pid': pid, 'rows': rows, 'context_pending': not own,
              'on_expected_gpu': bool(own) and all(r['gpu_uuid'] == uuid for r in own)}
    path = Path(directory) / 'gpu_processes.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record) + '\n')
    if (own and not record['on_expected_gpu']) or (require_present and not own):
        raise RuntimeError(f'{stage}: PID {pid} not exclusively on {uuid}')
    return record


def add_worker_args(parser, directory):
    parser.add_argument('--worker', action='store_true', help=__import__('argparse').SUPPRESS)
    parser.add_argument('--run-dir', type=Path, default=F / directory)


def guarded(script, directory):
    """Guard actual child launch and all runtime timing; retain contaminated runs."""
    import cpu_guard
    directory = Path(directory)
    require_memory()

    def launch(out):
        out.mkdir(parents=True, exist_ok=True)
        with cpu_guard.Sampler() as sampler:
            def factory():
                require_memory()
                return subprocess.Popen([sys.executable, str(script), *sys.argv[1:], '--worker', '--run-dir', str(out)], cwd=F)
            child = sampler.launch(factory)
            try:
                audit_gpu('worker immediately after launch', out, child.pid, require_present=False)
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.kill()
                child.wait()
                raise
        sampler.write(out / 'contamination.json')
        if code:
            raise RuntimeError(f'worker exited {code}; see {out}')
        return out
    return cpu_guard.guarded_run(launch, directory, attempts=3)
