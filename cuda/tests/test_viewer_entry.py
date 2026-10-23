"""Viewer entry points without opening any window: `python -m viewer --interactive` (a person at the PC) opens at once —
no idle / full-screen check, no waiting, only the GPU-memory rule; every script path (viewer --script,
viewer/launch.py used by tests, measure and make_media) keeps the desktop-protection checks; a bad --gpu index is
refused before any window or CUDA; the OpenGL context is destroyed at once if its renderer is not the GPU chosen for
CUDA; the parent-process GPU checks refuse on low FreeVirtualMemory. The checks are replaced by recording stand-ins;
nothing opens a window."""
import subprocess
import sys
import time

import pytest

from mytet_check import F
from mytet_cuda import device

sys.path.insert(0, str(F / "viewer"))
import app  # noqa: E402
import desktop  # noqa: E402
import launch  # noqa: E402
import render  # noqa: E402

SCRIPT = str(F / "viewer" / "scripts" / "P_negreal.json")
TI, GPU0 = "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 3090"  # names whose one is a prefix of the other
FAKE = {"index": 1, "uuid": "GPU-test", "name": TI, "free_mib": 23000, "used_mib": 850}


def spy(monkeypatch, gpu_free=23000, idle=0.0, fullscreen=False, memory_kb=50_000_000):
    calls, opened = [], []
    monkeypatch.setattr(desktop, "idle_seconds", lambda: calls.append("idle") or idle)
    monkeypatch.setattr(desktop, "fullscreen_app_in_front", lambda: calls.append("fullscreen") or fullscreen)
    monkeypatch.setattr(desktop, "free_virtual_kb", lambda: calls.append("memory") or memory_kb)
    monkeypatch.setattr(desktop, "may_open_window", lambda: calls.append("may_open_window") or (False, "user active"))
    monkeypatch.setattr(app, "gpu_free_mib", lambda: calls.append("gpu") or (FAKE, gpu_free))

    class StubViewer:
        def __init__(self, start, mode, readback, log):
            opened.append(mode)
            self.meta = {}

        def run(self, events):
            pass

        def save(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(app, "Viewer", StubViewer)
    return calls, opened


def test_interactive_entry_opens_without_desktop_checks(monkeypatch):
    calls, opened = spy(monkeypatch)
    rc = app.main(["--interactive"])
    assert not {"idle", "fullscreen", "may_open_window"} & set(calls), "SIG:interactive_no_desktop_checks " + str(calls)
    assert rc == 0 and opened == ["interactive"]
    assert "gpu" in calls  # the GPU-memory rule stays


def test_interactive_entry_refuses_without_gpu_room(monkeypatch):
    calls, opened = spy(monkeypatch, gpu_free=1000)
    assert app.main(["--interactive"]) == 3 and opened == []
    assert not {"idle", "fullscreen", "may_open_window"} & set(calls)


def test_interactive_entry_does_not_wait(monkeypatch):
    """From main() through the real Viewer constructor and the real render.Display constructor up to the call that
    creates the window (glfw.create_window): no sleep call on the way, well under 1 s. Replaced: nvidia-smi, the
    chosen GPU's name, glfw.init / window_hint (no GLFW state) and glfw.create_window, which records the moment it is
    reached and stops there (no window)."""
    waits, reached = [], []

    class Reached(Exception):
        pass

    def create_window(*a):
        reached.append(time.perf_counter())
        raise Reached

    monkeypatch.setattr(app, "gpu_free_mib", lambda: (FAKE, 23000))
    monkeypatch.setattr(app, "gpu_used_mib", lambda: 850)
    monkeypatch.setattr(render, "chosen_gpu_name", lambda: TI)
    monkeypatch.setattr(render, "machine_rule_applies", lambda: False)  # no local machine rule (same on every PC)
    monkeypatch.setattr(render.glfw, "init", lambda: True)
    monkeypatch.setattr(render.glfw, "window_hint", lambda *a: None)
    monkeypatch.setattr(render.glfw, "create_window", create_window)
    monkeypatch.setattr(time, "sleep", waits.append)  # records instead of sleeping
    t0 = time.perf_counter()
    try:
        rc = app.main(["--interactive"])
    except Reached:
        rc = "reached glfw.create_window"
    assert rc == "reached glfw.create_window" and len(reached) == 1 and waits == [] and reached[0] - t0 < 1.0, \
        f"SIG:interactive_waited rc {rc} sleeps {waits} reached {[r - t0 for r in reached]}"


def test_script_entry_keeps_desktop_checks(monkeypatch):
    calls, opened = spy(monkeypatch)  # the user was active 0 s ago
    rc = app.main(["--script", SCRIPT, "--mode", "timed"])
    assert {"idle", "fullscreen", "gpu"} <= set(calls), calls
    assert rc == 3 and opened == []


@pytest.mark.parametrize("case,kw", [("full-screen app in front", dict(idle=600.0, fullscreen=True)),
                                     ("FreeVirtualMemory too low", dict(idle=600.0, memory_kb=1_000_000)),
                                     ("user active", dict(idle=10.0))])
def test_script_entry_refuses_each_condition_alone(monkeypatch, case, kw):
    calls, opened = spy(monkeypatch, **kw)
    assert app.main(["--script", SCRIPT, "--mode", "timed"]) == 3 and opened == [], "SIG:script_guard " + case


def test_script_entry_opens_when_all_conditions_hold(monkeypatch):
    """Control for the refusals above: idle 600 s, no full-screen app, memory fine -> the viewer is created."""
    calls, opened = spy(monkeypatch, idle=600.0)
    assert app.main(["--script", SCRIPT, "--mode", "timed"]) == 0 and opened == ["timed"]


def test_launcher_keeps_desktop_checks(monkeypatch):
    calls, _ = spy(monkeypatch)
    monkeypatch.setattr(launch, "gpu_free_mib", lambda: 23000)
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    with pytest.raises(launch.NotNow):
        launch.wait_for_room(max_wait=0)
    assert "may_open_window" in calls


def test_script_and_interactive_exclude_each_other(monkeypatch):
    spy(monkeypatch)
    with pytest.raises(SystemExit):
        app.main(["--interactive", "--script", SCRIPT])


@pytest.mark.parametrize("err", [FileNotFoundError(2, "not found", "nvidia-smi"),
                                 subprocess.CalledProcessError(9, ["nvidia-smi"])])
def test_gpu_query_failure_is_a_one_line_refusal(monkeypatch, capsys, err):
    def fail(*a, **k):
        raise err
    monkeypatch.setattr(device.subprocess, "run", fail)
    assert app.main(["--interactive"]) == 3
    out = capsys.readouterr().err
    assert "not opening the window: cannot read the GPU list with nvidia-smi" in out and "Traceback" not in out
    assert len(out.strip().splitlines()) == 1


@pytest.mark.parametrize("bad", ["a", "-1", "1.0", "", "99"])
def test_bad_gpu_index_refused_before_any_window(monkeypatch, capsys, bad):
    """--gpu that is not a listed nvidia-smi index: one line, exit 2, no viewer, no window (real nvidia-smi query)."""
    calls, opened = spy(monkeypatch)
    assert app.main(["--interactive", "--gpu", bad]) == 2 and opened == [] and "gpu" not in calls
    out = capsys.readouterr().err.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("not opening the window: ")


class _Stop(Exception):
    pass


def _glfw_spies(monkeypatch, renderer):
    """GLFW / GL stand-ins that record the order of calls. create_window returns a fake handle (no window);
    the first GL call (glGetString(GL_RENDERER)) answers `renderer`; set_window_pos stops the constructor.
    No local machine rule (render.machine_rule_applies -> False), so the result is the same on every PC."""
    calls = []
    monkeypatch.setattr(render, "machine_rule_applies", lambda: False)

    def rec(name, ret=None):
        def f(*a, **k):
            calls.append(name)
            return ret
        return f

    def set_window_pos(*a):
        calls.append("set_window_pos")
        raise _Stop
    for name, f in (("init", rec("init", True)), ("window_hint", lambda *a: None),
                    ("create_window", rec("create_window", "fake-window")),
                    ("make_context_current", lambda w: calls.append(f"make_context_current({w})")),
                    ("destroy_window", rec("destroy_window")), ("terminate", rec("terminate")),
                    ("set_window_pos", set_window_pos), ("swap_interval", rec("swap_interval"))):
        monkeypatch.setattr(render.glfw, name, f)
    def gl_get_string(what):
        calls.append("glGetString")
        if isinstance(renderer, Exception):
            raise renderer
        return None if renderer is None else renderer.encode()
    monkeypatch.setattr(render.GL, "glGetString", gl_get_string)
    monkeypatch.setattr(render.GL, "glGenVertexArrays", rec("glGenVertexArrays", 1))
    monkeypatch.setattr(render.GL, "glGenTextures", rec("glGenTextures", 1))
    return calls


@pytest.mark.parametrize("chosen,renderer", [(TI, GPU0 + "/PCIe/SSE2"), (GPU0, TI + "/PCIe/SSE2"),
                                             (TI, "Intel(R) UHD Graphics 770"), (TI, ""), (TI, None),
                                             (TI, RuntimeError("GL error"))])
def test_gl_context_destroyed_at_once_if_the_renderer_is_not_the_chosen_gpu(monkeypatch, chosen, renderer):
    """The driver put the OpenGL context on another GPU than the one chosen for CUDA (also when one name is a prefix
    of the other): the renderer string is the first GL call, then the context and window are destroyed, before any
    drawing object or window move."""
    calls = _glfw_spies(monkeypatch, renderer)
    monkeypatch.setattr(render, "chosen_gpu_name", lambda: chosen)
    with pytest.raises((RuntimeError, _Stop)) as e:
        render.Display()
    want = ["init", "create_window", "make_context_current(fake-window)", "glGetString", "make_context_current(None)",
            "destroy_window", "terminate"]
    assert calls == want and "not the GPU chosen for CUDA" in str(e.value), f"SIG:gl_renderer_checked_first {calls}"


def test_gl_allowed_on_the_chosen_gpu(monkeypatch):
    """Control: renderer = the chosen GPU -> the constructor goes on past the check (to the window move, where the
    stand-in stops it)."""
    calls = _glfw_spies(monkeypatch, TI + "/PCIe/SSE2")
    monkeypatch.setattr(render, "chosen_gpu_name", lambda: TI)
    with pytest.raises(_Stop):
        render.Display()
    assert calls == ["init", "create_window", "make_context_current(fake-window)", "glGetString", "set_window_pos"]


@pytest.mark.parametrize("call", ["check_run", "check_display", "quality", "blocks"])
def test_parent_gpu_checks_refuse_on_low_memory(monkeypatch, call):
    """The parent-process GPU work of tests / measure (viewer_check) refuses when FreeVirtualMemory < 8,388,608 KB,
    before reading any run or touching the GPU (the log folder does not even exist)."""
    import viewer_check as vc
    monkeypatch.setattr(desktop, "free_virtual_kb", lambda: 8_388_607)
    with pytest.raises(vc.MemoryRefused):
        getattr(vc, call)(F / "test_out" / "no-such-run")
    monkeypatch.setattr(desktop, "free_virtual_kb", lambda: 8_388_608)  # control: enough memory -> goes on, then
    with pytest.raises(FileNotFoundError):                              # fails reading the missing run
        getattr(vc, call)(F / "test_out" / "no-such-run")
