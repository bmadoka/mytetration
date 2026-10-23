"""Four one-line mutations; CPU preflight: --check.

The normal CPU+GPU baseline must pass every test before any source is changed.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F/'scripts'))
sys.path.insert(0, str(F/'tests'))
from perturb_tools import add_worker_args, guarded, require_memory, write_json
from fault_judge import _is_intended, control_status, read_report

TESTS = ('tests/test_deepzoom2_engine_gpu.py', 'tests/test_deepzoom2_engine_cpu.py',
         'tests/test_perturb_gpu.py', 'tests/test_perturb_cpu.py')
# Tool self-tests use private copies and synthetic reports, not numerical fault evidence.
TEST_FILTER = 'not test_tools_'
FAULTS = {
    1: ('deep/reference.py', 'period = max(1, e-first-1)'),
    2: ('deep/perturb_kernel.cu', 'return true;'),
    3: ('deep/perturb_kernel.cu', 'return false;'),
    4: ('deep/perturb_kernel.cu', 'return false;'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def anchor(root, number):
    """Require one comment marker across all deep source files, on its expected line."""
    rel, replacement = FAULTS[number]
    pattern = re.compile(rf'(?:#|//)\s*FAULT_{number}\b')
    found = [(p, i, line) for p in (Path(root)/'deep').iterdir()
             if p.suffix in ('.py', '.cu')
             for i, line in enumerate(p.read_bytes().decode('utf-8').splitlines(keepends=True))
             if pattern.search(line)]
    if len(found) != 1 or found[0][0] != Path(root)/rel:
        raise ValueError(f'expected unique FAULT_{number} in {rel}: {len(found)} markers')
    path, index, line = found[0]
    indent = line[:len(line)-len(line.lstrip())]
    ending = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
    return path, index, line, indent+replacement+ending


@contextmanager
def mutation(root, number, entry):
    path, index, old, new = anchor(root, number)
    original = path.read_bytes()
    entry.update(fault=number, file=FAULTS[number][0], original_line=old.rstrip('\r\n'),
                 replacement=FAULTS[number][1], original_sha256=sha(original))
    try:
        lines = original.decode('utf-8').splitlines(keepends=True)
        lines[index] = new
        path.write_bytes(''.join(lines).encode('utf-8'))
        entry['mutated_sha256'] = sha(path.read_bytes())
        if entry['mutated_sha256'] == entry['original_sha256']:
            raise RuntimeError('mutation did not change bytes')
        yield
    finally:
        path.write_bytes(original)
        entry['restored_sha256'] = sha(path.read_bytes())
        entry['sha_restored'] = entry['restored_sha256'] == entry['original_sha256']
        if not entry['sha_restored']:
            raise RuntimeError(f'original SHA not restored: {path}')


def baseline_ok(code, rows):
    status = control_status(rows)
    calls = {r['nodeid'] for r in rows if r['when'] == 'call' and r['outcome'] == 'passed'}
    return (code == 0 and bool(status) and calls == set(status)
            and all(v == 'passed' for v in status.values())
            and not any(r['outcome'] != 'passed' or r.get('xfail') for r in rows))


def judge(code, rows, baseline):
    control = control_status(baseline)
    ran = {r['nodeid'] for r in rows if r['when'] != 'collect'}
    failed = [r for r in rows if r['outcome'] == 'failed']
    intended = [r for r in failed if _is_intended(r) and control.get(r['nodeid']) == 'passed']
    infra = [r for r in failed if r not in intended]
    omitted = [r for r in rows if r['outcome'] == 'skipped' or r.get('xfail')]
    caught = code == 1 and bool(intended) and not infra and not omitted and ran == set(control)
    return dict(caught=caught, assertion_failure_observed=bool(intended),
                same_tests=ran == set(control), infrastructure_failure=bool(infra),
                missing_tests=sorted(set(control)-ran), extra_tests=sorted(ran-set(control)),
                skipped_or_xfail=omitted,
                failures=[dict(test=r['nodeid'], phase=r['when'], exception=r.get('exc_type'),
                               raised_in_test_body=r.get('raised_in_test_body', False),
                               at_assert_statement=r.get('at_assert_statement', False),
                               message_first_line=r.get('message_first_line', ''),
                               assertion_error=r in intended) for r in failed])


def run_pytest(directory, label):
    require_memory()
    token = uuid.uuid4().hex
    records = directory/f'{label}_{token}.jsonl'
    env = dict(os.environ, MYTET_FAULT_REPORT=str(records.resolve()), MYTET_FAULT_RUN=token,
               PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', *TESTS, '-k', TEST_FILTER, '-q', '-rA',
                             '--basetemp', str(directory/f'{label}_tmp_{token}')],
                            cwd=F, env=env, capture_output=True, text=True, encoding='utf-8')
    (directory/f'{label}_stdout.txt').write_text(result.stdout, encoding='utf-8')
    (directory/f'{label}_stderr.txt').write_text(result.stderr, encoding='utf-8')
    return result.returncode, read_report(records, token) if records.exists() else []


def preflight(directory):
    """Exercise real replacement/restoration on private copies, including raised exceptions."""
    directory.mkdir(parents=True, exist_ok=True)
    copy_root = directory/('source_copy_'+uuid.uuid4().hex)
    (copy_root/'deep').mkdir(parents=True)
    for rel in {v[0] for v in FAULTS.values()}:
        shutil.copyfile(F/rel, copy_root/rel)
    report = dict(kind='CPU preflight only; no pytest fault detection claim', faults=[])
    for number in FAULTS:
        anchor(F, number)
        entry = {}
        try:
            with mutation(copy_root, number, entry):
                raise InterruptedError('exercise exceptional restoration')
        except InterruptedError:
            entry['exceptional_restore_exercised'] = True
        report['faults'].append(entry)
    write_json(directory/'preflight.json', report)
    return report


def run(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for number in FAULTS:
        anchor(F, number)
    report = dict(tests=list(TESTS), test_filter=TEST_FILTER, baseline=None, faults=[], all_faults_caught=False)
    try:
        code, rows = run_pytest(directory, 'baseline')
        report['baseline'] = dict(returncode=code, records=rows)
        write_json(directory/'faults.json', report)
        if not baseline_ok(code, rows):
            raise RuntimeError('normal run did not pass every test; no faults were inserted')
        for number in FAULTS:
            entry = {}
            try:
                with mutation(F, number, entry):
                    code, records = run_pytest(directory, f'fault_{number}')
                    entry.update(returncode=code, records=records, **judge(code, records, rows))
            finally:
                report['faults'].append(entry)
                write_json(directory/'faults.json', report)
        missed = [e['fault'] for e in report['faults'] if not (e.get('caught') and e.get('sha_restored'))]
        report['all_faults_caught'] = not missed and len(report['faults']) == len(FAULTS)
        if not report['all_faults_caught']:
            raise RuntimeError(f'faults not caught by test-body assertions: {missed}')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(directory/'faults.json', report)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    add_worker_args(parser, 'logs/deepzoom2/faults')
    parser.add_argument('--check', action='store_true', help='CPU-only copied-source restoration checks')
    args = parser.parse_args()
    if args.check:
        preflight(args.run_dir)
    elif args.worker:
        run(args.run_dir)
    else:
        guarded(Path(__file__), args.run_dir)


if __name__ == '__main__':
    main()
