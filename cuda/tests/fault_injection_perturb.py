"""One-line faults, structured pytest records, unconditional restoration.

Run from cuda/: .venv/Scripts/python.exe tests/fault_injection_perturb.py
The baseline must exit zero before any source mutation occurs.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

F=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(F/'scripts'))
from perturb_tools import add_worker_args, guarded, require_memory, write_json

FAULTS = {
    1: ('deep/perturb_kernel.cu','    return pt_mk(pt_exp(w).re-1.0,pt_exp(w).im); // FAULT_1'),
    2: ('deep/perturb_kernel.cu','        pt_c w=pt_mul(delta,a); // FAULT_2'),
    3: ('deep/perturb_kernel.cu','    return pt_naive_log1p(u); // FAULT_3'),
    4: ('deep/limits.py','    return depth_digits(eps) - 5  # FAULT_4'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_pytest(directory,label):
    require_memory()
    token=uuid.uuid4().hex
    records=directory/f'{label}_{token}.jsonl'
    env=dict(os.environ,MYTET_FAULT_REPORT=str(records),MYTET_FAULT_RUN=token,PYTHONDONTWRITEBYTECODE='1')
    # Every CUDA stage is audited by the GPU suite, including the sparse test.
    result=subprocess.run([sys.executable,'-B','-m','pytest','tests/test_perturb_gpu.py',
                           'tests/test_perturb_cpu.py','-q','-rA'],cwd=F,env=env,capture_output=True,text=True)
    (directory/f'{label}_stdout.txt').write_text(result.stdout,encoding='utf-8')
    (directory/f'{label}_stderr.txt').write_text(result.stderr,encoding='utf-8')
    rows=[json.loads(s) for s in records.read_text(encoding='utf-8').splitlines()] if records.exists() else []
    assert all(r.get('run')==token for r in rows), 'foreign pytest report records'
    return result.returncode,rows


def run(directory):
    directory.mkdir(parents=True,exist_ok=True)
    baseline_code,baseline=run_pytest(directory,'baseline')
    report={'baseline':{'returncode':baseline_code,'records':baseline},'faults':[]}
    write_json(directory/'faults.json',report)
    if baseline_code or not baseline or any(r['outcome'] in ('failed','skipped') for r in baseline):
        raise RuntimeError('normal run did not pass every test; no faults were inserted')
    baseline_tests={r['nodeid'] for r in baseline if r['when']=='call' and r['outcome']=='passed'}
    for number,(rel,replacement) in FAULTS.items():
        path=F/rel
        original=path.read_bytes()
        original_sha=sha(original)
        text=original.decode('utf-8')
        lines=text.splitlines(keepends=True)
        found=[i for i,line in enumerate(lines) if f'FAULT_{number}' in line]
        if len(found)!=1:
            raise RuntimeError(f'expected unique FAULT_{number} in {rel}')
        entry={'fault':number,'file':rel,'original_sha256':original_sha,'replacement':replacement}
        try:
            i=found[0]
            ending='\r\n' if lines[i].endswith('\r\n') else '\n'
            lines[i]=replacement+ending
            path.write_bytes(''.join(lines).encode('utf-8'))
            entry['mutated_sha256']=sha(path.read_bytes())
            code,records=run_pytest(directory,f'fault_{number}')
            failures=[r for r in records if r['outcome']=='failed']
            entry.update(returncode=code,failures=[{
                'test':r['nodeid'],'phase':r['when'],'exception':r.get('exc_type'),
                'assertion_error':r.get('exc_type')=='builtins.AssertionError',
                'message_first_line':r.get('message_first_line',''),
                'at_assert_statement':r.get('at_assert_statement',False)} for r in failures])
            assertion_failures=[r for r in failures if r['when']=='call' and
                                r.get('exc_type')=='builtins.AssertionError' and r.get('at_assert_statement')]
            infra=[r for r in failures if r not in assertion_failures]
            tested={r['nodeid'] for r in records if r['when']=='call'}
            entry.update(assertion_failure_observed=bool(assertion_failures),
                         infrastructure_failure=bool(infra),same_tests=tested==baseline_tests)
        finally:
            path.write_bytes(original)
            entry['restored_sha256']=sha(path.read_bytes())
            entry['sha_restored']=entry['restored_sha256']==original_sha
            report['faults'].append(entry)
            write_json(directory/'faults.json',report)
            if not entry['sha_restored']:
                raise RuntimeError(f'original SHA not restored: {path}')
    # A fault that is not caught by an assertion (same tests, no infrastructure failure)
    # makes the runner fail instead of returning normally.
    missed = [e['fault'] for e in report['faults']
              if not (e.get('assertion_failure_observed') and not e.get('infrastructure_failure') and e.get('same_tests'))]
    report['all_faults_caught'] = not missed and len(report['faults']) == len(FAULTS)
    write_json(directory/'faults.json',report)
    if not report['all_faults_caught']:
        raise SystemExit(f'faults not caught by assertions (or infrastructure failures): {missed}')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    add_worker_args(parser,'logs/perturb/faults')
    args=parser.parse_args()
    if args.worker:
        run(args.run_dir)
    else:
        guarded(Path(__file__),args.run_dir)


if __name__=='__main__':
    main()
