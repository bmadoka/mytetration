"""GPU choice (mytet_cuda.device, `--gpu N` / MYTET_GPU): the index maps to the nvidia-smi device of that index, and a
bad index is refused before CUDA is loaded. Nothing here starts CUDA on any GPU except `test_cuda_sees_only_the_chosen_gpu`,
which uses the GPU of this test session (MYTET_GPU)."""
import os
import subprocess
import sys

import pytest

from mytet_check import F
from mytet_cuda import device


def smi(index):
    """uuid, name of one index straight from nvidia-smi (independent of device.py's parsing)."""
    out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=uuid,name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    uuid, name = [s.strip() for s in out.strip().split(",", 1)]
    return uuid, name


def test_index_maps_to_the_nvidia_smi_device():
    gpus = device.list_gpus()
    assert gpus and [g["index"] for g in gpus] == list(range(len(gpus)))
    for g in gpus:
        c = device.choose(str(g["index"]))
        assert (c["index"], c["uuid"], c["name"]) == (g["index"], g["uuid"], g["name"])
        assert (c["uuid"], c["name"]) == smi(g["index"])


@pytest.mark.parametrize("bad", ["-1", "a", "1.0", "", " ", "+1", "0x1", "99"])
def test_bad_index_is_refused(bad):
    with pytest.raises(device.GpuChoiceError):
        device.choose(bad)


@pytest.mark.parametrize("bad", ["7", "-1", "a"])
def test_bad_index_refused_before_cuda_is_loaded(bad):
    """In a fresh process with MYTET_GPU=bad, importing mytet_cuda.gpu fails with GpuChoiceError and CuPy (CUDA) was
    never imported. Index 7 is only used when nvidia-smi lists fewer than 8 GPUs."""
    if bad == "7" and len(device.list_gpus()) > 7:
        pytest.skip("this machine has a GPU 7")
    code = ("import sys\nfrom mytet_cuda import device\ntry:\n    import mytet_cuda.gpu\nexcept device.GpuChoiceError:\n"
            "    print('refused', 'cupy' in sys.modules)\n")
    cp = subprocess.run([sys.executable, "-c", code], cwd=str(F), capture_output=True, text=True,
                        env=dict(os.environ, MYTET_GPU=bad), timeout=60)
    assert cp.stdout.strip() == "refused False", (cp.stdout, cp.stderr[-500:])


@pytest.mark.parametrize("bad", ["7", "-1", "a"])
def test_cli_bad_gpu_is_one_line_and_nonzero(run_dir, bad):
    if bad == "7" and len(device.list_gpus()) > 7:
        pytest.skip("this machine has a GPU 7")
    out = run_dir / f"cli_bad_gpu_{bad}"
    cp = subprocess.run([sys.executable, "-m", "mytet_cuda", "1by1_H", "--n", "9", "--gpu", bad, "--out-dir", str(out)],
                        cwd=str(F), capture_output=True, text=True, timeout=60)
    lines = cp.stderr.strip().splitlines()
    assert cp.returncode == 2 and len(lines) == 1 and "error:" in lines[0], cp.stderr
    assert not out.exists()  # stopped before any output


@pytest.mark.parametrize("bad", ["7", "-1", "a", "9" * 5000])
def test_cli_and_viewer_refuse_before_cupy_is_loaded(bad):
    """Fresh processes: the CLI and the viewer entry refuse a bad --gpu with exit 2 and CuPy (CUDA) / GLFW never
    imported."""
    if bad == "7" and len(device.list_gpus()) > 7:
        pytest.skip("this machine has a GPU 7")
    cli = ("import runpy, sys\nsys.argv = ['mytet_cuda', '1by1_H', '--n', '9', '--gpu', sys.argv[1]]\n"
           "try:\n    runpy.run_module('mytet_cuda', run_name='__main__')\nexcept SystemExit as e:\n"
           "    print('exit', e.code, 'cupy' in sys.modules)\n")
    viewer = ("import sys\nsys.path.insert(0, 'viewer')\nimport app\nrc = app.main(['--interactive', '--gpu', sys.argv[1]])\n"
              "print('exit', rc, 'cupy' in sys.modules or 'glfw' in sys.modules)\n")
    for code in (cli, viewer):
        cp = subprocess.run([sys.executable, "-c", code, bad], cwd=str(F), capture_output=True, text=True, timeout=60)
        assert cp.stdout.strip() == "exit 2 False", (cp.stdout, cp.stderr[-500:])
        assert len(cp.stderr.strip().splitlines()) == 1, cp.stderr[-500:]


def test_gpu_option_overrides_the_environment(run_dir):
    """--gpu (device.use) wins over a wrong MYTET_GPU in the same process and reaches CUDA and child processes: a fresh
    process with MYTET_GPU=a binds the session's GPU after device.use(), and the CLI with --gpu draws a picture."""
    want = device.choose()  # the session's GPU (MYTET_GPU)
    code = ("import os, sys\nfrom mytet_cuda import device\ndevice.use(sys.argv[1])\nimport mytet_cuda.gpu as g\n"
            "print(g.DEVICE['index'], os.environ['CUDA_VISIBLE_DEVICES'] == g.DEVICE['uuid'], os.environ['MYTET_GPU'])\n")
    cp = subprocess.run([sys.executable, "-c", code, str(want["index"])], cwd=str(F), capture_output=True, text=True,
                        env=dict(os.environ, MYTET_GPU="a"), timeout=60)
    assert cp.stdout.split() == [str(want["index"]), "True", str(want["index"])], (cp.stdout, cp.stderr[-500:])
    out = run_dir / "cli_gpu_option"
    cp = subprocess.run([sys.executable, "-m", "mytet_cuda", "1by1_H", "--n", "9", "--gpu", str(want["index"]),
                         "--out-dir", str(out)], cwd=str(F), capture_output=True, text=True, timeout=300,
                        env=dict(os.environ, MYTET_GPU="a"))
    assert cp.returncode == 0 and len(list(out.glob("*.png"))) == 1, cp.stderr[-800:]


def test_no_switch_after_cuda_is_bound(gpu, monkeypatch):
    """Once mytet_cuda.gpu is bound to a GPU in this process, choosing another one is refused (device.use), and a
    changed MYTET_GPU stops the next GPU call before any CUDA work (gpu._check_device). Needs a second GPU in the
    nvidia-smi list; the other GPU is only looked up, never used."""
    others = [g for g in device.list_gpus() if g["uuid"] != gpu.DEVICE["uuid"]]
    if not others:
        pytest.skip("only one GPU")
    with pytest.raises(device.GpuChoiceError):
        device.use(str(others[0]["index"]))
    assert device.choose()["index"] == gpu.DEVICE["index"]
    monkeypatch.setenv(device.GPU_ENV, str(others[0]["index"]))
    with pytest.raises(RuntimeError, match="bound to GPU"):
        gpu.onestep([2.0], [3.0])


def test_launcher_passes_the_choice_to_the_viewer_process(monkeypatch, run_dir):
    """viewer/launch.py starts the viewer with this process's environment, so the viewer uses the same GPU."""
    sys.path.insert(0, str(F / "viewer"))
    import launch
    seen = {}

    class FakePopen:  # nvidia-smi trace and the viewer process: record the viewer's environment instead of starting it
        returncode = 0

        def __init__(self, args, **k):
            if args[0] != "nvidia-smi":
                seen["env"] = k.get("env")

        def communicate(self, timeout=None):
            return "", ""

        def kill(self):
            pass

        def terminate(self):
            pass

        def wait(self, t=None):
            return 0
    g = device.choose()
    monkeypatch.setenv(device.GPU_ENV, str(g["index"]))  # the session's GPU, written as --gpu would
    monkeypatch.setattr(launch.device, "choose", lambda text=None: g)  # no nvidia-smi call through the stubs below
    monkeypatch.setattr(launch, "wait_for_room", lambda max_wait=0, log=None: {})
    monkeypatch.setattr(launch.subprocess, "Popen", FakePopen)

    class Sampler:  # the CPU guard of launch.run is not under test here
        wait_reasons = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def launch(self, factory):
            return factory()

        def write(self, path):
            pass
    monkeypatch.setattr(launch.cpu_guard, "Sampler", Sampler)
    launch.run(F / "viewer" / "scripts" / "P_negreal.json", "timed", run_dir / "launch_env")
    assert seen["env"][device.GPU_ENV] == os.environ[device.GPU_ENV]


def test_cuda_sees_only_the_chosen_gpu(gpu):
    """The session's GPU (MYTET_GPU): CUDA sees one device, the one nvidia-smi lists under that index."""
    want = device.choose()
    assert gpu.DEVICE["uuid"] == want["uuid"] == os.environ["CUDA_VISIBLE_DEVICES"]
    info = gpu.device_info()
    assert (info["index"], info["uuid"], info["name"]) == (want["index"], want["uuid"], want["name"])
    assert gpu.cp.cuda.runtime.getDeviceCount() == 1
