"""Structured «was the injected fault caught by its intended check?» judgement (exact exception type, failure
location, control baseline).

Recording (pytest plugin hooks below, active only when MYTET_FAULT_REPORT names a file; tests/conftest.py imports
them then): one JSON line per test phase and per failed collection, with the run token MYTET_FAULT_RUN, the outcome,
skip / xfail, and for a failure the exact exception class, whether it was raised in the frame of the test function
itself (code object identity: not a helper, not a nested function), whether that line is inside an `assert`
statement of the test file, and the first line of the exception message (never the traceback or quoted source).

Judgement, per fault run against the control run (same test files, no fault):
  intended  = phase «call», exception class exactly builtins.AssertionError, raised in the test function's own frame
              at an `assert` statement, and the first message line holds the intended tag as a whole token
              (e.g. «SIG:render_target», not part of another tag).
  infra     = every other failure: setup / teardown / collection errors, other exception classes (RuntimeError from a
              viewer launch, launch.NotNow, AssertionError subclasses), AssertionError from a helper or a bare `raise`,
              an assertion message naming a launch problem, an XPASS(strict).
  caught    = no infra failure, the fault run ran exactly the control's tests, and an intended failure is in a test
              that PASSED in the control (call phase passed, not xfail).
  undetermined (판정 불가) = an intended failure exists but its test was skipped / xfail / absent / failing in the
              control, or the test sets differ; never counted as caught.
"""
import ast
import json
import os
import re
from pathlib import Path

import pytest

INFRA_MARKERS = ("NotNow", "viewer exit", "TimeoutExpired", "not opening the window", "contaminated")
TAG = re.compile(r"(?<![A-Za-z0-9_:])SIG:[A-Za-z0-9_]+(?![A-Za-z0-9_])")
_ASSERT_LINES = {}


def _assert_lines(path):
    """Line numbers covered by `assert` statements in a source file."""
    if path not in _ASSERT_LINES:
        lines = set()
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        except (OSError, SyntaxError, ValueError):
            pass
        _ASSERT_LINES[path] = lines
    return _ASSERT_LINES[path]


def _write(rec):
    path = os.environ.get("MYTET_FAULT_REPORT")
    if path:
        rec["run"] = os.environ.get("MYTET_FAULT_RUN", "")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def record(item, call, report):
    rec = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
           "xfail": hasattr(report, "wasxfail")}
    ei = call.excinfo
    if ei is not None and report.outcome == "failed":
        t = ei.type
        tb = ei.tb
        while tb.tb_next is not None:
            tb = tb.tb_next
        fn = getattr(item, "function", None)
        raise_file, raise_line = tb.tb_frame.f_code.co_filename, tb.tb_lineno
        msg = str(ei.value)
        rec.update({"exc_type": f"{t.__module__}.{t.__qualname__}",
                    "raised_in_test_body": fn is not None and tb.tb_frame.f_code is getattr(fn, "__code__", None),
                    "raise_file": raise_file, "raise_line": raise_line,
                    "at_assert_statement": raise_line in _assert_lines(raise_file),
                    "message_first_line": msg.splitlines()[0] if msg else "",
                    "longrepr_tail": str(report.longrepr)[-1500:]})
    elif report.outcome == "failed":  # e.g. XPASS(strict): a failure without an exception
        rec.update({"exc_type": None, "longrepr_tail": str(report.longrepr)[-1500:]})
    _write(rec)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if os.environ.get("MYTET_FAULT_REPORT"):
        record(item, call, outcome.get_result())


def pytest_collectreport(report):
    if os.environ.get("MYTET_FAULT_REPORT") and report.outcome == "failed":
        _write({"nodeid": report.nodeid, "when": "collect", "outcome": "failed", "xfail": False, "exc_type": None,
                "longrepr_tail": str(report.longrepr)[-1500:]})


def read_report(path, token):
    """Records of one pytest run; every line must carry this run's token (no stale or foreign lines)."""
    recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    foreign = [r for r in recs if r.get("run") != token]
    if foreign:
        raise ValueError(f"{len(foreign)} report lines are not from run {token}")
    return recs


def _is_intended(r):
    return (r["when"] == "call" and r.get("exc_type") == "builtins.AssertionError" and r.get("raised_in_test_body") is True
            and r.get("at_assert_statement") is True and not any(m in r.get("message_first_line", "") for m in INFRA_MARKERS))


def control_status(recs):
    """nodeid -> «passed» / «skipped» / «xfail» / «failed» from the control run's records."""
    st = {}
    for r in recs:
        if r["when"] == "collect":
            continue
        cur = st.get(r["nodeid"], "passed")
        if r["outcome"] == "failed":
            cur = "failed"
        elif r["xfail"] and cur != "failed":
            cur = "xfail"
        elif r["outcome"] == "skipped" and cur == "passed":
            cur = "skipped"
        st[r["nodeid"]] = cur
    return st


def judge(fault_recs, control_recs, signal):
    ctl = control_status(control_recs)
    ran = {r["nodeid"] for r in fault_recs if r["when"] != "collect"}
    failed = [r for r in fault_recs if r["outcome"] == "failed"]
    infra, tags, uncertain = [], {}, []
    for r in failed:
        if not _is_intended(r):
            infra.append({"test": r["nodeid"], "when": r["when"], "exc_type": r.get("exc_type"),
                          "raised_in_test_body": r.get("raised_in_test_body"),
                          "at_assert_statement": r.get("at_assert_statement"),
                          "where": f"{Path(r['raise_file']).name}:{r['raise_line']}" if r.get("raise_file") else None,
                          "message": (r.get("message_first_line") or r.get("longrepr_tail", ""))[:300]})
            continue
        tags[r["nodeid"]] = sorted(set(TAG.findall(r["message_first_line"])))
    with_signal = [n for n, t in tags.items() if signal in t]
    valid = [n for n in with_signal if ctl.get(n) == "passed"]
    uncertain = [{"test": n, "control": ctl.get(n, "absent")} for n in with_signal if ctl.get(n) != "passed"]
    files = {n.split("::")[0] for n in ran}
    ctl_same_files = {n for n in ctl if n.split("::")[0] in files}
    same_tests = ran == ctl_same_files
    if infra:
        verdict = "not caught (infrastructure failure)"
    elif valid and same_tests:
        verdict = "caught"
    elif with_signal:
        verdict = "undetermined"
    else:
        verdict = "not caught"
    return {"caught": verdict == "caught", "verdict": verdict, "intended_signal": signal, "tests": len(ran),
            "failed_tests": sorted({r["nodeid"] for r in failed}), "signals_by_test": tags, "caught_by": valid,
            "undetermined": uncertain, "same_tests_as_control": same_tests,
            "missing_vs_control": sorted(ctl_same_files - ran), "extra_vs_control": sorted(ran - ctl_same_files),
            "infrastructure_failure": infra}
