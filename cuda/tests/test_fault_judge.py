"""The fault-injection judge (tests/fault_judge.py) counts a fault as caught only when a test that passed in the control
run fails with an exact AssertionError raised at an `assert` of the test function itself whose first message line holds
the intended tag. Regressions: the fault (f) log quotes the test source, which
contains «SIG:render_target»; judging that run as fault (e) must say «not caught»; and
(a helper's «AssertionError: SIG:params helper launch failed» must not be a catch; control skip / xfail -> undetermined;
a report line from another run is rejected). The last test records a real pytest run with the plugin hooks."""
import json
import os
import subprocess
import sys

import pytest

from fault_judge import judge, read_report
from mytet_check import F

F_LOG = (F / "tests" / "data" / "fault_f_stdout.txt").read_text(encoding="utf-8")
F_FIRST = "SIGS SIG:params SIG:precise_params SIG:precise_map SIG:precise_display"


def rec(nodeid, outcome="passed", when="call", **kw):
    """One report line as the plugin writes it; a failure defaults to an intended-looking assertion of the test body."""
    r = {"nodeid": nodeid, "when": when, "outcome": outcome, "xfail": False, "run": "t"}
    if outcome == "failed":
        r.update({"exc_type": "builtins.AssertionError", "raised_in_test_body": True, "raise_file": "tests/test_viewer.py",
                  "raise_line": 98, "at_assert_statement": True, "message_first_line": F_FIRST, "longrepr_tail": ""})
    r.update(kw)
    return r


def control(*ids):
    return [rec(n) for n in ids]


A, B, C = "tests/test_viewer.py::test_a", "tests/test_viewer.py::test_b", "tests/test_viewer.py::test_c"


def test_quoted_source_is_not_a_signal():
    assert "SIG:render_target" in F_LOG  # the old stdout search found this (quoted test source) -> false «caught»
    run = [rec(A, "failed", longrepr_tail=F_LOG), rec(B, "failed", longrepr_tail=F_LOG), rec(C)]
    e = judge(run, control(A, B, C), "SIG:render_target")
    assert e["caught"] is False and e["failed_tests"] and not e["infrastructure_failure"]
    assert judge(run, control(A, B, C), "SIG:params")["caught"] is True


def test_tags_are_whole_tokens_and_first_line_only():
    run = [rec(A, "failed", message_first_line="SIGS SIG:precise_params", longrepr_tail="assert ['SIG:params']")]
    assert judge(run, control(A), "SIG:params")["caught"] is False  # only inside another tag / not on the first line
    assert judge(run, control(A), "SIG:precise_params")["caught"] is True


@pytest.mark.parametrize("bad", [
    dict(message_first_line="SIG:params helper launch failed", raised_in_test_body=False, raise_line=36),  # the probe
    dict(exc_type="builtins.RuntimeError", message_first_line="viewer exit 3: not opening the window: SIG:params"),
    dict(exc_type="launch.NotNow", message_first_line="13:00 waiting SIG:params"),
    dict(exc_type="test_viewer.Sub", message_first_line="SIGS SIG:params"),                 # AssertionError subclass
    dict(at_assert_statement=False, message_first_line="SIGS SIG:params"),                   # bare `raise`
    dict(message_first_line="SIGS SIG:params (viewer exit 1)"),
    dict(when="setup"), dict(when="teardown"),
    dict(when="collect", exc_type=None, message_first_line="")])
def test_infrastructure_failures_are_not_catches(bad):
    """Each of these alone, and next to a genuine catch in another test, gives «not caught (infrastructure failure)»."""
    for run in ([rec(B, "failed", **bad), rec(A)], [rec(B, "failed", **bad), rec(A, "failed")]):
        r = judge(run, control(A, B), "SIG:params")
        assert r["caught"] is False and r["infrastructure_failure"] and r["verdict"].startswith("not caught"), (bad, r)


@pytest.mark.parametrize("ctl_line,status", [(dict(outcome="skipped"), "skipped"), (dict(outcome="skipped", xfail=True), "xfail"),
                                             (dict(outcome="failed"), "failed"), (None, "absent")])
def test_control_must_have_passed_the_catching_test(ctl_line, status):
    ctl = [rec(B)] + ([rec(A, **ctl_line)] if ctl_line else [])
    run = [rec(A, "failed", message_first_line="SIGS SIG:params"), rec(B)]
    r = judge(run, ctl, "SIG:params")
    assert r["caught"] is False and r["verdict"] == "undetermined" and r["undetermined"] == [{"test": A, "control": status}]


def test_test_set_must_match_the_control():
    run = [rec(A, "failed", message_first_line="SIGS SIG:params")]  # test_b of the control did not run
    r = judge(run, control(A, B), "SIG:params")
    assert r["caught"] is False and r["missing_vs_control"] == [B]


def test_no_failure_is_not_a_catch():
    assert judge([rec(A)], control(A), "SIG:params")["verdict"] == "not caught"


def test_report_lines_of_another_run_are_rejected(run_dir):
    p = run_dir / "fault_judge_token.jsonl"
    p.write_text(json.dumps(rec(A)) + "\n" + json.dumps({**rec(B), "run": "old"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_report(p, "t")
    p.write_text(json.dumps(rec(A)) + "\n", encoding="utf-8")  # control: only this run's lines
    assert len(read_report(p, "t")) == 1


PROBE = '''import pytest


def helper():
    raise AssertionError("SIG:params helper launch failed")


def helper_assert():
    assert False, "SIGS SIG:params from a helper assert"


class Sub(AssertionError):
    pass


def test_body():
    v = ["SIG:params"]
    assert not v, "SIGS " + " ".join(v)


def test_helper():
    helper()


def test_helper_assert():
    helper_assert()


def test_nested():
    def inner():
        raise AssertionError("SIGS SIG:params nested")
    inner()


def test_bare_raise():
    raise AssertionError("SIGS SIG:params")


def test_subclass():
    raise Sub("SIGS SIG:params")


def test_runtime():
    raise RuntimeError("AssertionError: SIGS SIG:params")


def test_pass():
    assert True


@pytest.mark.skip(reason="probe")
def test_skipped():
    pass


@pytest.mark.xfail(strict=True)
def test_xfail():
    assert False, "SIGS SIG:params"


@pytest.fixture
def broken():
    assert False, "SIGS SIG:params in setup"


def test_setup_error(broken):
    pass
'''


def test_plugin_records_a_real_pytest_run(run_dir):
    """A real pytest run (CPU only, subprocess) with the plugin hooks: exact exception class, test-body frame, assert
    line, first message line, skip / xfail; then the probe counterexample is «not caught»."""
    d = run_dir / "fault_judge_probe"
    d.mkdir()
    (d / "conftest.py").write_text(f"import sys\nsys.path.insert(0, {str(F / 'tests')!r})\n"
                                   "from fault_judge import pytest_collectreport, pytest_runtest_makereport  # noqa\n",
                                   encoding="utf-8")
    (d / "test_probe.py").write_text(PROBE, encoding="utf-8")
    rep = d / "report.jsonl"
    env = dict(os.environ, MYTET_FAULT_REPORT=str(rep), MYTET_FAULT_RUN="probe-run")
    cp = subprocess.run([sys.executable, "-m", "pytest", "-q", "-W", "ignore", "test_probe.py"], cwd=str(d), env=env,
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert cp.returncode == 1, cp.stdout[-2000:]
    recs = read_report(rep, "probe-run")
    by = {(r["nodeid"].split("::")[1], r["when"]): r for r in recs}
    t = "builtins.AssertionError"
    want = {  # test: (exc_type, raised_in_test_body, at_assert_statement)
        "test_body": (t, True, True), "test_helper": (t, False, False), "test_helper_assert": (t, False, True),
        "test_nested": (t, False, False), "test_bare_raise": (t, True, False),
        "test_subclass": ("test_probe.Sub", True, False), "test_runtime": ("builtins.RuntimeError", True, False)}
    got = {n: (by[(n, "call")]["exc_type"], by[(n, "call")]["raised_in_test_body"], by[(n, "call")]["at_assert_statement"])
           for n in want}
    assert got == want
    assert by[("test_body", "call")]["message_first_line"] == "SIGS SIG:params"
    assert by[("test_helper", "call")]["message_first_line"] == "SIG:params helper launch failed"
    assert by[("test_pass", "call")]["outcome"] == "passed"
    assert by[("test_skipped", "setup")]["outcome"] == "skipped" and ("test_skipped", "call") not in by
    assert by[("test_xfail", "call")]["outcome"] == "skipped" and by[("test_xfail", "call")]["xfail"] is True
    assert by[("test_setup_error", "setup")]["outcome"] == "failed"
    assert by[("test_setup_error", "setup")]["raised_in_test_body"] is False
    ctl = [{**r, "outcome": "passed", "xfail": False} for r in recs]  # a control where every test passed
    pick = lambda *names: [r for r in recs if r["nodeid"].split("::")[1] in names]  # noqa: E731
    ctl_of = lambda *names: [r for r in ctl if r["nodeid"].split("::")[1] in names]  # noqa: E731
    probe = judge(pick("test_helper", "test_pass"), ctl_of("test_helper", "test_pass"), "SIG:params")
    assert probe["caught"] is False and probe["infrastructure_failure"][0]["test"].endswith("::test_helper")
    assert judge(pick("test_body", "test_pass"), ctl_of("test_body", "test_pass"), "SIG:params")["caught"] is True
    assert judge(recs, ctl, "SIG:params")["caught"] is False  # helper / setup / RuntimeError failures present
    ctl_skip = [r for r in recs if r["nodeid"].split("::")[1] in ("test_skipped", "test_pass")]
    moved = [{**r, "nodeid": r["nodeid"].replace("test_body", "test_skipped")} for r in pick("test_body", "test_pass")]
    assert judge(moved, ctl_skip, "SIG:params")["verdict"] == "undetermined"


def test_fault_selection_control_is_exact_union_and_each_judge_gets_its_subset(monkeypatch):
    import fault_injection_viewer as fi
    collected = []
    def collect(args):
        collected.append(args)
        return {"tests/test_hybrid_cpu.py::test_a"} if "-k" in args else {"tests/test_hybrid_cpu.py::test_b"}
    monkeypatch.setattr(fi, "collect_selection", collect)
    choices = {"j": ["tests/test_hybrid_cpu.py", "-k", "test_a"],
               "l": ["tests/test_hybrid_cpu.py::test_b"]}
    selected, nodes, union = fi.selection_plan(["j", "l"], choices)
    assert selected == choices and collected == list(choices.values())
    assert union == ["tests/test_hybrid_cpu.py::test_a", "tests/test_hybrid_cpu.py::test_b"]
    assert nodes["l"] == {"tests/test_hybrid_cpu.py::test_b"}
    with pytest.raises(ValueError, match="unknown"):
        fi.selection_plan(["l"], {"unknown": []})


def test_fault_j_and_l_have_distinct_live_anchors():
    import fault_injection_viewer as fi
    for key in ("j", "l", "window_wait"):
        _, path, old, new, _, signal = fi.FAULTS[key]
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1 and old != new
    assert fi.FAULTS["l"][-1] == "SIG:duplicate_jobs"
    assert fi.FAULTS["j"][-1] == "SIG:fresh_replacement"
    assert "if self.fresh and False" not in fi.APP.read_text(encoding="utf-8")


@pytest.mark.parametrize("option", [["-x"], ["-qx"], ["--exitfirst"], ["--maxfail=1"], ["--maxfail", "1"]])
def test_fault_selections_reject_early_stop_before_collection(monkeypatch, option):
    import fault_injection_viewer as fi
    def no_subprocess(*args, **kwargs):
        pytest.fail("early-stop selection reached subprocess")
    monkeypatch.setattr(fi.subprocess, "run", no_subprocess)
    with pytest.raises(ValueError, match="early-stop"):
        fi.collect_selection(["tests/test_hybrid_cpu.py::test_switching_frame_does_not_start_duplicate_job",
                              "tests/test_realtime_video.py", *option])


def test_fault_cross_file_maxfail_input_is_undetermined(tmp_path):
    """First file fails the intended assertion; --maxfail=1 skips the second file entirely."""
    import fault_injection_viewer as fi
    (tmp_path / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(F / 'tests')!r})\n"
        "from fault_judge import pytest_collectreport, pytest_runtest_makereport\n", encoding="utf-8")
    (tmp_path / "test_switching.py").write_text(
        'def test_duplicate():\n    assert False, "SIG:duplicate_jobs"\n', encoding="utf-8")
    (tmp_path / "test_other.py").write_text('def test_other():\n    assert True\n', encoding="utf-8")
    rep = tmp_path / "maxfail.report.jsonl"
    env = dict(os.environ, MYTET_FAULT_REPORT=str(rep), MYTET_FAULT_RUN="maxfail-review")
    cp = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         "test_switching.py", "test_other.py", "--maxfail=1"], cwd=tmp_path, env=env,
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert cp.returncode == 1, cp.stdout + cp.stderr
    recs = read_report(rep, "maxfail-review")
    first = next(r["nodeid"] for r in recs if r["when"] == "call")
    other = first.replace("test_switching.py::test_duplicate", "test_other.py::test_other")
    ctl = control(first, other)
    # The old file-filtering judge alone reproduces the false catch.
    assert judge(recs, ctl, "SIG:duplicate_jobs")["caught"] is True
    result = fi.judge_selection(recs, ctl, "SIG:duplicate_jobs", {first, other})
    assert result["caught"] is False and result["verdict"] == "undetermined"
    assert result["execution"]["missing_intended"] == [other]
    assert result["same_tests_as_control"] is False and result["missing_vs_control"] == [other]


@pytest.mark.parametrize("phase", [None, "setup", "collect"])
@pytest.mark.parametrize("incomplete", ["fault", "control"])
def test_fault_selection_requires_call_phase_for_every_intended_node(phase, incomplete):
    import fault_injection_viewer as fi
    other = "tests/test_other.py::test_other"
    fault = [rec(A, "failed", message_first_line="SIG:duplicate_jobs"), rec(other)]
    ctl = control(A, other)
    short = [r for r in (fault if incomplete == "fault" else ctl) if r["nodeid"] != other]
    if phase:
        short.append(rec(other, "failed", when=phase))
    result = fi.judge_selection(short if incomplete == "fault" else fault,
                                short if incomplete == "control" else ctl, "SIG:duplicate_jobs", {A, other})
    assert result["caught"] is False and result["verdict"] == "undetermined"
    key = "execution" if incomplete == "fault" else "control_execution"
    assert result[key]["missing_intended"] == [other]


def test_fault_run_checks_intended_subset_after_execution_and_restores(monkeypatch, tmp_path):
    import fault_injection_viewer as fi
    from types import SimpleNamespace
    source = tmp_path / "fault_target.py"
    source.write_bytes(b"original\r\n")
    monkeypatch.setattr(fi, "FAULTS", {"probe": ("CPU probe", source, "original", "mutated", "unused", "SIG:duplicate_jobs")})
    monkeypatch.setattr(fi, "LOGS", tmp_path)
    other = "tests/test_other.py::test_other"
    cp = SimpleNamespace(returncode=1, stdout="1 failed", stderr="")
    monkeypatch.setattr(fi, "pytest", lambda *args: (cp, tmp_path / "probe.report.jsonl",
                        [rec(A, "failed", message_first_line="SIG:duplicate_jobs")], True, 0.))
    result = fi.run("probe", "", control(A, other), selection=[A, other], nodeids={A, other})
    assert result["caught"] is False and result["verdict"] == "undetermined"
    assert result["execution"]["missing_intended"] == [other]
    assert result["restored_ok"] and source.read_bytes() == b"original\r\n"


def test_fault_main_reports_incomplete_control_as_undetermined(monkeypatch, tmp_path):
    import fault_injection_viewer as fi
    from types import SimpleNamespace
    out = tmp_path / "control_summary.json"
    monkeypatch.setattr(sys, "argv", ["fault_injection_viewer.py", "--logs", str(tmp_path), "--out", str(out), "l"])
    monkeypatch.setattr(fi, "selection_plan", lambda *args: ({"l": [A, B]}, {"l": {A, B}}, [A, B]))
    cp = SimpleNamespace(returncode=0, stdout="1 passed", stderr="")
    monkeypatch.setattr(fi, "pytest", lambda *args: (cp, tmp_path / "control.report.jsonl", control(A), True, 0.))
    def no_fault(*args, **kwargs):
        pytest.fail("incomplete control reached fault mutation")
    monkeypatch.setattr(fi, "run", no_fault)
    with pytest.raises(SystemExit, match="control run undetermined"):
        fi.main()
    result = json.loads(out.read_text(encoding="utf-8"))[0]
    assert result["verdict"] == "undetermined" and result["execution"]["missing_intended"] == [B]
