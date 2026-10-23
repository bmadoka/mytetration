"""Chosen GPU only: accuracy, structure, state parity, and sparse/stride contracts.

No synthetic truth is substituted if fixture generation was blocked. Generate the
fixture with scripts/deepzoom2_eval.py --make-fixture once guards permit GPU work.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from deep import cpu_ref,hp,limits,perturb,reference,truth

F = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(F/'scripts'))
from perturb_tools import audit_gpu
from deepzoom2_eval import structure


@pytest.fixture(scope='module',autouse=True)
def audit_engine_context(gpu,run_dir):
    gpu.cp.empty(1)
    audit_gpu('deepzoom2 engine tests CUDA context',run_dir)


@pytest.fixture(scope='module')
def old_maps(gpu,run_dir):
    rows = json.loads((F/'tests/data/perturb_truth_fixture.json').read_text(encoding='utf-8'))['positions']
    maps = []
    for row in rows:
        assert row['max_iter']==500
        m,info = perturb.deep_map(row['x0'],row['y0'],row['eps'],max_iter=row['max_iter'])
        maps.append((row,m,info))
    return maps


def test_accuracy_stage1_fixture_zero_mismatch(old_maps):
    assert len(old_maps)==12
    for row,m,info in old_maps:
        assert not info['out_of_range'],(row['id'],row['eps'],info['reason'])
        actual = np.asarray([m[x,y] for x,y in truth.SAMPLE])
        assert np.count_nonzero(actual!=row['diverged'])==0,(row['id'],row['eps'])


@pytest.fixture(scope='module')
def engine_truth_fixture():
    path = F/'tests/data/deepzoom2_truth_fixture.json'
    assert path.exists(),'deepzoom2 truth fixture has not been generated'
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data.get('version')==2
    rows = data['positions']
    assert {(r['id'],r['eps']) for r in rows}=={('m37',1e-20),('m37',1e-40)},data.get('blocked_reason')
    for row in rows:
        assert row['max_iter']==limits.max_iter_for(row['eps'])
        assert len(row['groups']['band']['pixels'])>=1500
        assert len(row['groups']['uniform']['pixels'])>=500
    assert len(data['n6']['pixels'])==61*61
    return data


@pytest.fixture(scope='module')
def m37_maps(engine_truth_fixture,gpu):
    values = []
    for row in engine_truth_fixture['positions']:
        prepared = perturb.plan(row['x0'],row['y0'],row['eps'],max_iter=row['max_iter'])
        job = perturb.launch(prepared)
        assert isinstance(job.done(),bool)
        m,info = job.result(return_stop=True)
        assert job.done()
        values.append((row,prepared,m,info))
    return values


def test_accuracy_and_structure_m37(m37_maps):
    for row,prepared,m,info in m37_maps:
        assert not info['out_of_range'],info['reason']
        metric = structure(m)
        assert metric['boundary_pairs']>=27500,metric
        assert metric['tile_ratio']>=.11,metric
        for name in ('sample625','uniform','band'):
            group = row['groups'][name]
            points = group['pixels']; targets = group['truth']
            assert len(points)==len(targets)
            indices = [i for i,v in enumerate(targets) if v['decided']]
            assert len(indices)/len(points)>=.95,(name,'insufficient decided truth')
            assert sum(v['unstable'] for v in targets)/len(points)<=.005,(name,'unstable truth')
            mismatch = sum(bool(m[tuple(points[i])])!=targets[i]['diverged'] for i in indices)
            assert mismatch/len(indices)<=.01,(row['eps'],name,mismatch,len(indices))
            stop_mismatch = sum(int(info['stop'][tuple(points[i])])!=targets[i]['stop'] for i in indices)
            assert stop_mismatch/len(indices)<=.01,(row['eps'],name,'stop',stop_mismatch,len(indices))


def test_samples_and_strides_equal_final(m37_maps,gpu):
    for row,prepared,m,info in m37_maps:
        pixels = row['groups']['uniform']['pixels'][:32]+row['groups']['band']['pixels'][:32]
        values,meta = perturb.deep_samples(row['x0'],row['y0'],row['eps'],pixels,max_iter=row['max_iter'])
        assert not meta['out_of_range'] and meta['refs'][0]['pixel']==info['refs'][0]['pixel']
        assert np.array_equal(values,[m[x,y] for x,y in pixels])
        assert np.array_equal(meta['stop'],[info['stop'][x,y] for x,y in pixels])
        for stride in (2,4):
            stream = gpu.cp.cuda.Stream(non_blocking=True)
            job = perturb.launch(prepared,stride=stride,stream=stream)
            coarse,coarse_info = job.result(return_stop=True)
            assert not coarse_info['out_of_range'],coarse_info['reason']
            assert np.array_equal(coarse,m[::stride,::stride])
            assert np.array_equal(coarse_info['stop'],info['stop'][::stride,::stride])


def test_cpu_device_verdict_and_stop_match(m37_maps):
    for row,prepared,m,info in m37_maps:
        pixels = row['groups']['uniform']['pixels'][:20]
        for x,y in pixels:
            value = cpu_ref.pixel(prepared.ref,complex(prepared.gx[x],prepared.gy[y]),row['max_iter'])
            assert value==(bool(m[x,y]),0,int(info['stop'][x,y])),((x,y),value,info['stop'][x,y])


def test_accuracy_n6_counterexample_and_61_square(engine_truth_fixture):
    row = engine_truth_fixture['n6']
    assert row['max_iter']==limits.max_iter_for(row['eps'])
    m,info = perturb.deep_map(row['x0'],row['y0'],row['eps'],max_iter=row['max_iter'],return_stop=True)
    pixels,targets = row['pixels'],row['truth']
    index = pixels.index([953,543])
    assert targets[index]['decided'],'counterexample must have a ladder truth decision'
    if not info['out_of_range']:
        assert bool(m[953,543])==targets[index]['diverged']
        assert int(info['stop'][953,543])==targets[index]['stop']
    decided = [i for i,v in enumerate(targets) if v['decided']]
    mismatch = sum(bool(m[tuple(pixels[i])])!=targets[i]['diverged'] for i in decided)
    assert len(decided)/len(pixels)>=.95
    assert sum(v['unstable'] for v in targets)/len(pixels)<=.005
    assert info['out_of_range'] or mismatch/len(decided)<=.01,('n6 window mismatch',mismatch,len(decided))


def test_small_uncertainty_uses_self_reference(gpu,monkeypatch):
    # Inject a log-preserved next reference on the uncertainty boundary. One
    # sampled pixel must fall back to its own HP orbit, whose answer is known.
    prepared = perturb.plan('1','0',1e-20,max_iter=3)
    prepared.ref.z[3]=0j
    prepared.ref.log_abs[3]=np.log(1e10)
    prepared.ref.log_abs_lo[3]=0.
    job = perturb.Job(prepared,1,None,np.asarray([960*1080+540],dtype=np.int64))
    out,info = job.result(return_stop=True)
    assert not info['out_of_range'] and not out[0]
    assert info['fallback_pixels']==1 and info['stop']==[3]


def test_many_uncertainties_report_unresolved(gpu):
    prepared = perturb.plan('1','0',1e-20,max_iter=3)
    prepared.ref.z[3]=0j
    prepared.ref.log_abs[3]=np.log(1e10)
    prepared.ref.log_abs_lo[3]=0.
    pixels = np.arange(65,dtype=np.int64)
    job = perturb.Job(prepared,1,None,pixels)
    _,info = job.result()
    assert info['out_of_range'] and info['reason']=='unresolved'
    assert info['fallback_pixels']==0 and info['flag_counts']['3']==65
