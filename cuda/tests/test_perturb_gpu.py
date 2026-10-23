"""CUDA checks, using conftest.gpu's device/memory rules.

Fixture schema: JSON list (or {"positions": [...]}) of objects with x0,y0 decimal
strings, eps, max_iter, digits, and diverged:625 booleans (truth also accepted).
Missing or empty fixtures are assertion failures, never skips.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from deep import hp, reference, truth
from deep import perturb
from deepzoom_blocks import compare
from perturb_cases import INPUTS, assert_components

F = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(F/'scripts'))
from perturb_tools import audit_gpu
from perturb_eval import independent_pixel


@pytest.fixture(scope='module', autouse=True)
def audit_test_context(gpu,run_dir):
    gpu.cp.empty(1)
    audit_gpu('perturb tests CUDA context',run_dir)


@pytest.fixture(scope='session')
def truth_fixture():
    path = F/'tests/data/perturb_truth_fixture.json'
    assert path.exists(), '픽스처 없음: tests/data/perturb_truth_fixture.json'
    rows = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(rows,dict):
        rows = rows.get('positions',rows.get('cases',[]))
    assert rows, '픽스처 없음: nonempty cases required'
    for row in rows:
        for key in ('x0','y0','eps','max_iter','digits'):
            assert key in row, f'fixture missing {key}'
        assert isinstance(row['x0'],str) and isinstance(row['y0'],str), 'decimal coordinate strings required'
        values = row.get('diverged',row.get('truth'))
        assert isinstance(values,list) and len(values)==625, 'fixture must contain 625 verdicts in SAMPLE order'
        assert all(isinstance(v,bool) for v in values), 'fixture verdicts must be bool'
        row['diverged'] = values
    return rows


def test_truth_fixture_present(truth_fixture):
    assert len(truth_fixture)>0


@pytest.fixture(scope='session')
def maps(truth_fixture,gpu,run_dir):
    result=[]
    for row in truth_fixture:
        m, info = perturb.deep_map(row['x0'],row['y0'],row['eps'],max_iter=row['max_iter'])
        audit_gpu('test deep_map',run_dir)
        result.append((row,m,info))
    return result


@pytest.mark.parametrize('name',['cexpm1','clog1p'])
def test_device_functions(gpu,run_dir,name):
    cp=gpu.cp
    gpu._check_device()
    em,lp = cp.empty(len(INPUTS),dtype=cp.complex128),cp.empty(len(INPUTS),dtype=cp.complex128)
    kernel = perturb._module(test=True).get_function('pt_functions')
    kernel(((len(INPUTS)+gpu.THREADS-1)//gpu.THREADS,),(gpu.THREADS,),
           (cp.asarray(INPUTS),np.int32(len(INPUTS)),em,lp))
    got = (em if name=='cexpm1' else lp).get()
    audit_gpu(f'test device {name}',run_dir)
    for z,value in zip(INPUTS,got):
        assert_components(value,z,name)


def test_accuracy(maps):
    for row,m,info in maps:
        expected=np.asarray(row['diverged'],dtype=bool)
        got=np.asarray([m[p,q] for p,q in truth.SAMPLE])
        count=int(np.count_nonzero(got!=expected))
        assert not info['out_of_range'], f"range: {info['reason']} at {row['x0']}, eps={row['eps']}"
        assert count/625<=.01, f"sample mismatch {count}/625 at {row['x0']}, eps={row['eps']}"


def test_unique_coordinates(maps):
    for row,m,info in maps:
        assert info['unique_x']==1920, f"unique_x={info['unique_x']} eps={row['eps']}"
        assert info['unique_y']==1080, f"unique_y={info['unique_y']} eps={row['eps']}"


def test_independent_reference_blocks(maps,gpu,run_dir):
    for row,m,info in maps:
        x,y,e,n = row['x0'],row['y0'],row['eps'],row['max_iter']
        pixel=independent_pixel(x,y,e,n,info['ref_digits']+40,info['refs'][0]['pixel'])
        other,meta=perturb.deep_map(x,y,e,max_iter=n,ref_digits=info['ref_digits']+40,ref_pixel=pixel)
        audit_gpu('test independent map',run_dir)
        blocks=compare(m,other)
        assert tuple(pixel)!=tuple(info['refs'][0]['pixel']), 'independent reference must differ'
        assert not meta['out_of_range'], f"second map range: {meta['reason']}"
        assert blocks['target_tiles']>=50, f"target tiles {blocks['target_tiles']} < 50"
        assert blocks['ok'], f'block comparison: {blocks}'


def test_sparse_matches_map(maps,gpu,run_dir):
    for row,m,info in maps:
        sampled,meta=perturb.deep_samples(row['x0'],row['y0'],row['eps'],truth.SAMPLE,max_iter=row['max_iter'])
        audit_gpu('test deep_samples',run_dir)
        assert not meta['out_of_range'], f"samples range: {meta['reason']}"
        assert np.array_equal(sampled,[m[p,q] for p,q in truth.SAMPLE]), 'deep_samples differs from deep_map'


def test_reference_out_of_range(gpu,monkeypatch):
    ref=reference.at_pixel('1','0',1e-20,(960,540),3,60)
    ref.log_abs[3]=np.nan
    monkeypatch.setattr(reference,'choose',lambda *a,**k:(ref,ref.seconds,[]))
    m,info=perturb.deep_map('1','0',1e-20,max_iter=3)
    assert info['out_of_range'] and info['reason']=='reference_exponent'
    assert info['reference_exponent']['n']==1
    assert not m.any()


def test_pixel_subnormal_range(gpu,run_dir,monkeypatch):
    monkeypatch.setattr(reference,'offsets',lambda *a:(np.full(1920,1e-310),np.zeros(1080)))
    result,info=perturb.deep_samples('1','0',1e-20,[(960,540)],max_iter=2)
    audit_gpu('test subnormal pixel',run_dir)
    assert info['out_of_range'] and info['reason']=='pixel_exponent'
    assert info['flag_counts']['2']==1


def test_initial_delta_and_escape_stop(gpu,run_dir):
    result,info=perturb.deep_samples('2','0',1e-20,[(960,540),(100,200)],max_iter=5)
    audit_gpu('test initial delta escape',run_dir)
    assert result.all() and not info['out_of_range']
    assert info['stop']==[4,4], f"stop numbers {info.get('stop')}"


def test_f64_coordinate_collapse(gpu):
    from mytet_cuda import variants
    x,y,_=variants.grid('16by9_H',1.4,.2,1e-20,1920)
    assert len(np.unique(x))<10, f'f64 unique_x={len(np.unique(x))}'


def test_deep_drag_recomputed(gpu,run_dir):
    from scripts.perturb_scripts import positions
    from perturb_tools import write_json
    p = positions()[0]
    before = hp.from_strings(p['x0'],p['y0'],1e-40)
    after = hp.apply_drag(before,dx=-100,dy=0)
    old, old_info = perturb.deep_map(*before)
    audit_gpu('deep drag old',run_dir)
    new, new_info = perturb.deep_map(*after)
    audit_gpu('deep drag new',run_dir)
    rates = {k: float(np.count_nonzero(new[:1920-k,:] == old[k:,:])/((1920-k)*1080))
             for k in range(95,106)}
    rec = {'position': p, 'before': hp.to_record(before), 'after': hp.to_record(after),
           'match_rates': rates, 'old_info': old_info, 'new_info': new_info,
           'fresh_evidence': {'different_reference_centers': bool(old_info['refs'] and new_info['refs'] and
                              old_info['refs'][0]['center'] != new_info['refs'][0]['center']),
                              'old_compute_ms': old_info['t_ms']['total'], 'new_compute_ms': new_info['t_ms']['total']}}
    # Every run keeps its own record; only unmutated runs publish the shared evidence file
    write_json(run_dir/'deep_drag.json',rec)
    if not os.environ.get('MYTET_FAULT_RUN'):
        write_json(F/'logs/perturb/impl2_deep_drag.json',rec)
    assert not old_info['out_of_range'] and not new_info['out_of_range'], 'drag range exit'
    assert rates[100] >= .98, f'drag k=100 agreement {rates[100]}'
    assert all(rates[100] > v for k,v in rates.items() if k != 100), f'drag unique maximum: {rates}'
    assert rec['fresh_evidence']['different_reference_centers'], 'drag must change actual reference centre strings'
    assert old_info['t_ms']['total'] > 0 and new_info['t_ms']['total'] > 0
