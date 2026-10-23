"""Fault injection: put one deliberate fault into mytet_cuda/kernel.cu, run the whole test suite,
expect failures, restore the file and confirm its sha256. One fault per category, each run separately.

usage: python tests/fault_injection.py [FAULT ...]   (default: F1 F2 F3)
Writes logs/fault_<FAULT>.txt (full pytest output) and logs/fault_injection_summary.json.
"""
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

F = Path(__file__).resolve().parents[1]
KERNEL = F / "mytet_cuda" / "kernel.cu"
ESC = "if (tet_cabs(z.re, z.im) > escape_radius) {"
FAULTS = {
    "F1": ("① 판정·매개변수: escape_radius 입력 대신 고정값 1e10으로 판정", ESC,
           "if (tet_cabs(z.re, z.im) > 1e10) {"),
    "F2": ("② NaN·inf: NaN을 발산으로 셈 (!(|z| <= r))", ESC,
           "if (!(tet_cabs(z.re, z.im) <= escape_radius)) {"),
    "F3": ("③ 거듭제곱 특수 경우: 밑 0에서 b.re > 0 을 b.re >= 0 으로", "if (b.re > 0.0) return tet_mk(0.0, 0.0);",
           "if (b.re >= 0.0) return tet_mk(0.0, 0.0);"),
    "F4": ("③(추가) 정수 지수 분기: |n| < 100 을 < 99 로 (n = 99가 일반 경로로)",
           "if (b.im == 0.0 && b.re > -100.0 && b.re < 100.0) {",
           "if (b.im == 0.0 && b.re > -99.0 && b.re < 99.0) {"),
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run(fault):
    label, old, new = FAULTS[fault]
    orig = KERNEL.read_bytes()
    h0 = sha(orig)
    text = orig.decode("utf-8")
    assert text.count(old) == 1, (fault, text.count(old))
    mutated = text.replace(old, new).encode("utf-8")
    t0 = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        KERNEL.write_bytes(mutated)
        h1 = sha(KERNEL.read_bytes())
        assert h1 != h0
        cp = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rf", "tests"], cwd=str(F),
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        KERNEL.write_bytes(orig)
    h2 = sha(KERNEL.read_bytes())
    out = cp.stdout + "\n--- stderr ---\n" + cp.stderr
    (F / "logs" / f"fault_{fault}.txt").write_text(
        f"fault {fault}: {label}\n- {old}\n+ {new}\nstart {t0}\nkernel sha256 original {h0}\n"
        f"kernel sha256 with fault {h1}\nkernel sha256 after restore {h2}\npytest returncode {cp.returncode}\n\n" + out,
        encoding="utf-8")
    failed = [ln.split(" ", 1)[1].split(" - ")[0] for ln in cp.stdout.splitlines() if ln.startswith("FAILED ")]
    tail = [ln for ln in cp.stdout.splitlines() if " passed" in ln or " failed" in ln][-1:]
    return {"fault": fault, "label": label, "pytest_returncode": cp.returncode, "caught": cp.returncode == 1 and bool(failed),
            "failed_tests": failed, "summary": tail, "sha_original": h0, "sha_fault": h1, "sha_restored": h2,
            "restored_ok": h2 == h0}


def main():
    faults = sys.argv[1:] or ["F1", "F2", "F3"]
    res = [run(f) for f in faults]
    p = F / "logs" / "fault_injection_summary.json"
    prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    p.write_text(json.dumps(prev + res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in res:
        print(r["fault"], "caught" if r["caught"] else "NOT CAUGHT", r["summary"], "restored" if r["restored_ok"] else "RESTORE FAILED",
              len(r["failed_tests"]), "failed tests")
    sys.exit(0 if all(r["caught"] and r["restored_ok"] for r in res) else 1)


if __name__ == "__main__":
    main()
