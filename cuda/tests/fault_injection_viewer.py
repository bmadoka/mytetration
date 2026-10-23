"""Fault injection for the viewer: one deliberate fault at
a time, run the fault's test file, judge the result, restore the file and confirm its sha256.

usage: python tests/fault_injection_viewer.py [--out JSON] [--tag T] [a b c ...]   (default: every fault)
A clean control run of the same test files must pass first. Every pytest run writes a structured report
(tests/fault_judge.py plugin, via tests/conftest.py) to a file name unique to the run (time + pid + fault) that must not
exist before; it is used only if it was written after the run started and every line carries the run's random token.
«Caught» (tests/fault_judge.py): a test that passed in the control fails with an exact AssertionError at an `assert` of
the test function itself, the intended tag in the first message line, no infrastructure failure, same tests as the
control. Logs: logs/preview/fault_<x><tag>_<stamp>.txt + .report.jsonl, fault_control<tag>_<stamp>.*.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F / "tests"))
from fault_judge import control_status, judge, read_report  # noqa: E402

APP = F / "viewer" / "app.py"
FAST = F / "viewer" / "fast.py"
RENDER = F / "viewer" / "render.py"
LOGS = F / "logs" / "preview"
FRAME_MAP = "            m = (self.fast.fast_map if kind == \"fast\" else self.fast.precise_map)(*p)\n"
CREATE = "        self.win = glfw.create_window(WIN_W, WIN_H, title, None, None)\n"
VIEWER_TESTS, ENTRY_TESTS = "tests/test_viewer.py", "tests/test_viewer_entry.py"
STAMP = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
# key: (label, file, old text, new text, test file, intended signal)
FAULTS = {
    "a": ("멈춰도 정밀로 안 바뀜: 정밀 프레임도 빠른 함수로 그림", APP, FRAME_MAP,
          "            m = self.fast.fast_map(*p)\n", VIEWER_TESTS, "SIG:precise_map"),
    "b": ("정밀 그림이 기존과 다름: max_iter 499", FAST,
          "params.NX, params.MAX_ITER, params.ESCAPE_RADIUS)",
          "params.NX, params.MAX_ITER - 1, params.ESCAPE_RADIUS)", VIEWER_TESTS, "SIG:precise_map"),
    "c": ("정밀 프레임이 마지막 입력 앞의 매개변수로 그려짐", APP, "        p = self.state\n",
          "        p = self.prev_state if kind == \"precise\" else self.state\n", VIEWER_TESTS, "SIG:precise_params"),
    "d": ("빠른 프레임이 앞 프레임 맵을 다시 씀(두 장에 한 번)", APP, FRAME_MAP,
          "            m = self._last_map if (kind == \"fast\" and getattr(self, \"_last_map\", None) is not None "
          "and len(self.frames) % 2) else (self.fast.fast_map if kind == \"fast\" else self.fast.precise_map)(*p)\n"
          "            self._last_map = m\n", VIEWER_TESTS, "SIG:fresh_compute"),
    "e": ("빠른 표시 경로에서 색을 뒤집음", APP, "        self.disp.draw(m)\n",
          "        self.disp.draw(~m if kind == \"fast\" else m)\n", VIEWER_TESTS, "SIG:render_target"),
    "f": ("입력 이벤트 하나(번호 100)를 매개변수에 반영하지 않음", APP,
          "        self.state, self.hp, clamped = deep_state.apply_event(self.state, self.hp, ev, logged)\n",
          "        self.state, self.hp, clamped = (self.state, self.hp, False) if self.applied == 99 else deep_state.apply_event(self.state, self.hp, ev, logged)\n",
          VIEWER_TESTS, "SIG:params"),
    "interactive_guard": ("대화식 진입이 다시 유휴·전체 화면 검사를 부름(시험 경로 사전 점검을 씀)", APP,
          "    why = preflight() if a.script else preflight_interactive()\n", "    why = preflight()\n",
          ENTRY_TESTS, "SIG:interactive_no_desktop_checks"),
    "callback_drag": ("콜백 드래그: 커서 이동 뒤 _drag_from 갱신 삭제(연속 이동이 누른 점부터 재짐)", APP,
          "            self._drag_from = (x, y)\n", "", VIEWER_TESTS, "SIG:callback_inputs"),
    "g": ("Force float32 selection in deep zoom", FAST,
          'def choose_precision(x0, y0, eps):\n',
          'def choose_precision(x0, y0, eps):\n    return "f32"\n', VIEWER_TESTS, "SIG:coordinates"),
    # g2: c and log c stay double (exact coordinates); only the orbit state z is
    # rounded to float32 after every step.
    "g2": ("Keep double coordinates and log c, orbit z stored in float32 (never pt32)", FAST,
           '    # G2_FAULT_TARGET: plain f32 orbit with double coordinates, never perturbation.\n    precision = choose_precision(x0, y0, eps) if precision is None else precision\n',
           '    global SOURCE\n    if "G2_MUTATED" not in SOURCE:\n        SOURCE = SOURCE.replace(\'    dc lc = d_clog(c.re, c.im);\\n\', \'    dc lc = d_clog(c.re, c.im);\\n    fc zf = fmk((float)c.re, (float)c.im);\\n    unsigned char df = 0;\\n    for (int k = 0; k < max_iter; ++k) {\\n        dc zn = d_cpow(c, dmk((double)zf.re, (double)zf.im), lc);\\n        zf = fmk((float)zn.re, (float)zn.im);\\n        if (zn.re * zn.re + zn.im * zn.im > escape_radius * escape_radius || isinf(zn.re) || isinf(zn.im)) { df = 1; break; }\\n    }\\n    out[i * (long long)ny + j] = df;\\n    return; // G2_MUTATED\\n\')\n        _modules.clear()\n    precision = "f64"  # actual double c and log c; orbit rounded to float32 each step; never pt32\n', VIEWER_TESTS, "SIG:blocks"),
    "h": ("Disable interpolation, jump straight to target", F / "viewer/params.py",
          'def smooth_step(display, target, dt):\n',
          'def smooth_step(display, target, dt):\n    return tuple(target)\n', VIEWER_TESTS, "SIG:smooth_zoom"),
    "interactive_wait": ("대화식 진입에 대기 삽입(time.sleep 0.2 s)", APP, "    return gpu_room_reason()\n",
          "    time.sleep(0.2)\n    return gpu_room_reason()\n", ENTRY_TESTS, "SIG:interactive_waited"),
    "callback_origin": ("콜백: _cursor 초기값 (0, 0)(첫 누름이 GLFW 조회 없이 원점에서 잼)", APP, "        self._cursor = None\n",
          "        self._cursor = (0.0, 0.0)\n", VIEWER_TESTS, "SIG:callback_inputs"),
    "window_wait": ("창 생성 직전 대기 삽입(Display 안, glfw.create_window 앞 time.sleep 0.2 s)", RENDER, CREATE,
          "        __import__(\"time\").sleep(0.2)\n" + CREATE, ENTRY_TESTS, "SIG:interactive_waited"),
    "gl_renderer": ("GL_RENDERER 확인 끔(문맥이 다른 GPU에 생겨도 계속 감)", RENDER,
          "        if renderer_device(self.renderer) != self.gpu_name:\n", "        if False:\n", ENTRY_TESTS,
          "SIG:gl_renderer_checked_first"),
    "i": ("Hybrid off: never enter warped", F / "viewer/hybrid.py",
          'elif previous == "fresh-only" and m > 0.033:  # HYBRID_ENTER',
          'elif False:  # HYBRID_ENTER', VIEWER_TESTS, "SIG:hybrid_speed"),
    "j": ("Fresh replacement stops after first map", APP,
          '        # HYBRID_REPLACEMENT_ANCHOR\n',
          '        # HYBRID_REPLACEMENT_ANCHOR\n        if self.fresh:\n            return\n', VIEWER_TESTS, "SIG:fresh_replacement"),
    "k": ("Enter warped even in shallow scenes", F / "viewer/hybrid.py",
          'elif previous == "fresh-only" and m > 0.033:  # HYBRID_ENTER',
          'elif previous == "fresh-only":  # HYBRID_ENTER', VIEWER_TESTS, "SIG:hybrid_mode"),
    "l": ("Duplicate same-parameter job at motion start", APP,
          '        if hybrid.duplicate_params(p, computing, self.fresh):  # HYBRID_DUPLICATE_GUARD',
          '        if False:  # HYBRID_DUPLICATE_GUARD', VIEWER_TESTS, "SIG:duplicate_jobs"),
    "m": ("Margin ignored: use edge instead", F / "viewer/hybrid.py",
          '        usable = rec.get("margin_t_end") is not None and rec["margin_t_end"] < t_decide  # MARGIN_USABLE\n',
          '        usable = False  # MARGIN_USABLE\n', "tests/test_hybrid_b3_cpu.py::test_live_margin_record", "SIG:margin_ignored"),
    "n": ("SHA back before present", APP,
          '        rec["present_hr"] = self.disp.present()\n',
          '        key = ("fresh",rec["source_fresh_id"]) if kind=="fast" else ("precise",rec["precise_id"])\n'
          '        self._archive().shown(key,time.perf_counter())\n'
          '        self.map_archive.entries[key]["future"].result()\n'
          '        rec["present_hr"] = self.disp.present()\n',
          "tests/test_hybrid_b3_cpu.py::test_live_sha_record", "SIG:sha_on_path"),

}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pytest(files, name):
    """One pytest run with a fresh structured report: (completed process, report path, records, fresh?)."""
    rep = LOGS / f"{name}_{STAMP}.report.jsonl"
    if rep.exists():
        raise RuntimeError(f"{rep} exists before the run")
    token = uuid.uuid4().hex
    env = dict(os.environ, MYTET_FAULT_REPORT=str(rep), MYTET_FAULT_RUN=token)
    t_start = time.time()
    cp = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-rf", "-W", "ignore", "-p", "no:cacheprovider", *files], cwd=str(F), env=env,
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    fresh = rep.exists() and rep.stat().st_mtime >= t_start
    return cp, rep, (read_report(rep, token) if fresh else []), fresh, t_start


def run(key, tag, ctl_recs, selection=None, nodeids=None):
    label, path, old, new, test_file, signal = FAULTS[key]
    selection = [test_file] if selection is None else selection
    reject_early_stop(selection)
    nodeids = collect_selection(selection) if nodeids is None else set(nodeids)
    ctl_recs = [r for r in ctl_recs if r["nodeid"] in nodeids]
    orig = path.read_bytes()
    h0 = sha(orig)
    text = orig.decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"  # follow the file's line endings
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert text.count(old) == 1, (key, text.count(old))
    t0 = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        h1 = sha(path.read_bytes())
        cp, rep, recs, fresh, t_start = pytest(selection, f"fault_{key}{tag}")
    finally:
        path.write_bytes(orig)
    h2 = sha(path.read_bytes())
    log = LOGS / f"fault_{key}{tag}_{STAMP}.txt"
    log.write_text(
        f"fault ({key}): {label}\nfile {path.relative_to(F) if path.is_relative_to(F) else path}\n- {old}+ {new}\nstart {t0}\nsha256 original {h0}\n"
        f"sha256 with fault {h1}\nsha256 after restore {h2}\npytest returncode {cp.returncode}\nreport {rep.name} "
        f"(fresh {fresh})\n\n{cp.stdout}\n--- stderr ---\n{cp.stderr[-5000:]}", encoding="utf-8")
    j = judge_selection(recs, ctl_recs, signal, nodeids) if fresh else {
        "caught": False, "verdict": "undetermined", "intended_signal": signal,
        "missing_intended": sorted(nodeids), "reason": "no fresh report"}
    tail = [ln for ln in cp.stdout.splitlines() if " passed" in ln or " failed" in ln][-1:]
    return {"fault": key, "label": label, "file": str(path.relative_to(F) if path.is_relative_to(F) else path), "test_file": test_file,
            "selection": selection, "control_nodeids": sorted(nodeids) if nodeids is not None else None,
            "pytest_returncode": cp.returncode, "summary": tail, "log": log.name, "report": rep.name,
            "report_fresh": fresh, **j, "caught": bool(j["caught"] and cp.returncode == 1),
            "sha_original": h0, "sha_fault": h1, "sha_restored": h2, "restored_ok": h2 == h0}


def execution_coverage(recs, nodeids):
    """Only a call-phase record proves that an intended test body ran."""
    ran = {r["nodeid"] for r in recs if r["when"] == "call"}
    intended = set(nodeids)
    return {"complete": ran == intended, "executed_nodeids": sorted(ran),
            "missing_intended": sorted(intended - ran), "extra_executed": sorted(ran - intended)}


def judge_selection(recs, ctl_recs, signal, nodeids):
    result = judge(recs, ctl_recs, signal)
    fault, control = execution_coverage(recs, nodeids), execution_coverage(ctl_recs, nodeids)
    result.update(execution=fault, control_execution=control)
    if not fault["complete"] or not control["complete"]:
        result.update(caught=False, verdict="undetermined", same_tests_as_control=False,
                      missing_vs_control=sorted(set(control["executed_nodeids"]) - set(fault["executed_nodeids"])))
    return result


def reject_early_stop(args):
    for arg in args:
        if (arg.split("=", 1)[0] in ("--maxfail", "--exitfirst")
                or (arg.startswith("-") and not arg.startswith("--") and "x" in arg[1:])):
            raise ValueError(f"early-stop pytest option is not allowed: {arg}")


def collect_selection(args):
    """Resolve node ids / -k to the exact intended test set before any mutation."""
    reject_early_stop(args)
    cp = subprocess.run([sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
                        cwd=str(F), capture_output=True, text=True, encoding="utf-8", errors="replace")
    nodes = {line.strip() for line in cp.stdout.splitlines() if "::" in line and not line.startswith(" ")}
    if cp.returncode != 0 or not nodes:
        raise RuntimeError(f"selection collection failed: {args}: {cp.stdout} {cp.stderr}")
    return nodes


def selection_plan(keys, overrides):
    unknown = set(overrides) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown fault selection keys: {sorted(unknown)}")
    selections, nodes = {}, {}
    for key in keys:
        args = overrides.get(key, [FAULTS[key][4]])
        if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args):
            raise ValueError(f"selection for {key} must be a nonempty pytest argument list")
        selections[key] = args
        nodes[key] = collect_selection(args)
    return selections, nodes, sorted(set().union(*nodes.values()))


def main():
    global LOGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="summary JSON (default inside --logs)")
    ap.add_argument("--tag", default="", help="suffix for the per-fault log files")
    ap.add_argument("--selections", type=Path, help='JSON fault -> pytest arguments, e.g. {"l":["tests/test_viewer.py","-k","hybrid"]}')
    ap.add_argument("--logs", type=Path, help="directory for control and fault reports")
    ap.add_argument("keys", nargs="*")
    a = ap.parse_args()
    keys = a.keys or list(FAULTS)
    if a.logs is not None:
        LOGS = a.logs
    LOGS.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(a.selections.read_text(encoding="utf-8")) if a.selections else {}
    selections, nodeids, files = selection_plan(keys, overrides)
    cp, rep, ctl_recs, fresh, _ = pytest(files, f"fault_control{a.tag}")
    (LOGS / f"fault_control{a.tag}_{STAMP}.txt").write_text(cp.stdout + "\n--- stderr ---\n" + cp.stderr[-5000:],
                                                             encoding="utf-8")
    st = control_status(ctl_recs)
    coverage = execution_coverage(ctl_recs, files)
    ctl = {"fault": "control (no fault)", "test_files": files, "pytest_returncode": cp.returncode, "report": rep.name,
           "report_fresh": fresh, "execution": coverage, "verdict": "control" if coverage["complete"] else "undetermined",
           "summary": [ln for ln in cp.stdout.splitlines() if " passed" in ln or " failed" in ln][-1:],
           "tests_by_status": {s: sum(1 for v in st.values() if v == s) for s in sorted(set(st.values()))},
           "sha": {p.name: sha(p.read_bytes()) for p in (APP, FAST, RENDER, F / "viewer/params.py", F / "viewer/fast_kernel.cu")}}
    if cp.returncode != 0 or not fresh or not st or not coverage["complete"]:
        ctl["verdict"] = "undetermined"
        Path(a.out or LOGS / f"fault_injection_viewer_{STAMP}.json").write_text(
            json.dumps([ctl], indent=1, ensure_ascii=False), encoding="utf-8")
        sys.exit(f"control run undetermined: {ctl}")
    res = [ctl] + [run(k, a.tag, ctl_recs, selections[k], nodeids[k]) for k in keys]
    Path(a.out or LOGS / f"fault_injection_viewer_{STAMP}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("control", ctl["summary"], ctl["tests_by_status"])
    for r in res[1:]:
        print(r["fault"], r.get("verdict"), r["intended_signal"], r["summary"], r.get("caught_by"),
              "restored" if r["restored_ok"] else "RESTORE FAILED", r.get("infrastructure_failure"))
    sys.exit(0 if all(r["caught"] and r["restored_ok"] for r in res[1:]) else 1)


if __name__ == "__main__":
    main()
