"""Guarded deep-zoom evaluation, precision ladders and reusable truth fixtures.

Examples (F/.venv python):
  scripts/deepzoom2_eval.py --out logs/deepzoom2/eval
  scripts/deepzoom2_eval.py --depths 1e-20 1e-40 --uniform 500 --band 1500 \
      --make-fixture tests/data/deepzoom2_truth_fixture.json --out logs/deepzoom2/fixture
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb_tools import (F,add_worker_args,audit_gpu,finite_json,guarded,
                           require_memory,write_json)
from deep import hp,limits,truth

DEPTHS = [1e-15,1e-20,1e-25,1e-30,1e-35,1e-40]
DEFAULT_NAMES = ['m37','m191','m59','m60b']
N6 = {'id':'n6','x0':'0.37610494333279680704622167329529150248935039028424858371281725',
      'y0':'1.9671747211895914'}


def positions_from(path=None):
    source = F/'tests/data/deepzoom2_positions.json' if path is None else Path(path)
    data = json.loads(source.read_text(encoding='utf-8'))
    if isinstance(data,dict) and 'positions' in data:
        data = data['positions']
    if isinstance(data,dict):
        names = DEFAULT_NAMES if path is None else list(data)
        data = [dict(data[name],id=name) for name in names]
    for row in data:
        row.setdefault('id',row.get('name'))
        if not row['id'] or not isinstance(row['x0'],str) or not isinstance(row['y0'],str):
            raise ValueError('positions require name/id and decimal x0/y0 strings')
    return data


def structure(m):
    """A cross-tile pair belongs to its lower coordinate endpoint's tile."""
    if np.shape(m)!=(1920,1080):
        raise ValueError('structure requires final 1920 x 1080 map')
    tiles = np.zeros((96,54),dtype=bool)
    counts = []
    for axis in (0,1):
        x,y = np.nonzero(np.diff(m,axis=axis))
        counts.append(len(x))
        tiles[x//20,y//20] = True
    return {'horizontal_pairs':counts[0],'vertical_pairs':counts[1],
            'boundary_pairs':sum(counts),'boundary_tiles':int(tiles.sum()),
            'total_tiles':5184,'tile_ratio':float(tiles.mean()),
            'pair_tile_rule':'lower-coordinate endpoint'}


def population_stats(m,pixels,ladder):
    decisions = ladder['points']
    decided = [i for i,v in enumerate(decisions) if v['decided']]
    mismatch = sum(bool(m[pixels[i][0],pixels[i][1]])!=decisions[i]['diverged'] for i in decided)
    unstable = sum(v['unstable'] for v in decisions)
    total = len(pixels)
    divergent = sum(decisions[i]['diverged'] for i in decided)
    return {'sampled':total,'decided':len(decided),
            'decided_ratio':len(decided)/total if total else None,
            'decided_at_least_95pct':bool(total and len(decided)/total>=.95),
            'mismatch':mismatch,'mismatch_ratio':mismatch/len(decided) if decided else None,
            'mismatch_at_most_1pct':bool(decided and mismatch/len(decided)<=.01),
            'unstable':unstable,'unstable_ratio':unstable/total if total else None,
            'unstable_at_most_half_pct':bool(total and unstable/total<=.005),
            'diverged_fraction':divergent/len(decided) if decided else None,
            'nondiverged_fraction':1-divergent/len(decided) if decided else None,
            'truth_seconds':ladder['seconds'],'points_per_second':ladder['points_per_second'],
            'computed_verdicts':ladder['computed_verdicts'],'cached_verdicts':ladder['cached_verdicts'],
            'd3_points':ladder['d3_points']}


def compare_n2n(a,b):
    pairs = [(x,y) for x,y in zip(a['points'],b['points']) if x['decided'] and y['decided']]
    changed = sum(x['diverged']!=y['diverged'] for x,y in pairs)
    return {'both_decided':len(pairs),'changed':changed,'ratio':changed/len(pairs) if pairs else None}


def map_png(m,path):
    from PIL import Image
    Image.fromarray((m.T[::-1].astype(np.uint8)*255)).save(path)


def initial_png(positions,directory):
    sys.path.insert(0,str(F/'viewer'))  # viewer/fast.py imports `params` as a top-level module
    from viewer.fast import precise_map
    from PIL import Image,ImageDraw
    m = precise_map(0.,0.,5.)
    if isinstance(m,tuple):
        m = m[0]
    image = Image.fromarray(m.T[::-1].astype(np.uint8)*255).convert('RGB')
    draw = ImageDraw.Draw(image)
    for row in positions:
        x,y = float(row['x0']),float(row['y0'])
        px,py = (x+5)/10*1919,(5*9/16-y)/(10*9/16)*1079
        draw.ellipse((px-5,py-5,px+5,py+5),outline='red',width=2)
        draw.text((px+8,py-8),row['id'],fill='red')
    image.save(directory/'initial.png')


def save(args,rows,reduction):
    write_json(args.out/'eval.json',finite_json({'rows':rows,'reduction':reduction,
               'seeds':'SHA256(name|eps.hex) first 64 bits, little endian',
               'band_definition':'all transitions, lower endpoint +/-3 along transition axis',
               'limitations':'Samples cannot prove every pixel; N/2N cannot exclude later escapes. Viewer latency not measured.'}))
    lines = ['|position|eps|N|pairs|tile %|range|625 mismatch/decided|uniform mismatch/decided|band mismatch/decided|unstable (625/u/b)|N/2N changes (625/u)|total ms|',
             '|---|---:|---:|---:|---:|---|---|---|---|---|---|---:|']
    for row in rows:
        groups = row['groups']
        cells = [str(groups[k]['mismatch'])+'/'+str(groups[k]['decided']) for k in ('sample625','uniform','band')]
        unstable = '/'.join(str(groups[k]['unstable']) for k in ('sample625','uniform','band'))
        changed = '/'.join(str(row['n2n'][k]['ratio']) for k in ('sample625','uniform'))
        lines.append(f"|{row['id']}|{row['eps']:g}|{row['max_iter']}|{row['structure']['boundary_pairs']}|{100*row['structure']['tile_ratio']:.3f}|{row['info']['reason']}|"+'|'.join(cells)+f"|{unstable}|{changed}|{row['info']['t_ms']['total']:.2f}|")
    lines += ['', 'Times require the accompanying contamination.json. Cross-tile pairs belong to the lower endpoint tile.',
              'Sample sizes, undecided rates and range reasons must be inspected with mismatch rates.']
    (args.out/'eval_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def fixture_row(row,populations,ladders):
    return {'id':row['id'],'x0':row['x0'],'y0':row['y0'],'eps':row['eps'],
            'max_iter':row['max_iter'],'depth_digits':hp.depth_digits(row['eps']),
            'map_sha256':row['map_sha256'],'structure':row['structure'],'groups':{
                name:{'pixels':populations[name],'truth':ladders[name]['points']}
                for name in populations},'seed':row['seed']}


def n6_fixture(args):
    eps,n = 1e-15,limits.max_iter_for(1e-15)
    pixels = [(x,y) for x in range(923,984) for y in range(513,574)]
    target = truth.ladder(truth.pixel_points(N6['x0'],N6['y0'],eps,pixels),n,15,args.processes,args.truth_cache)
    return dict(N6,eps=eps,max_iter=n,pixels=pixels,truth=target['points'])


def run(args):
    require_memory()
    from deep import perturb
    from mytet_cuda import gpu
    gpu._check_device()
    gpu.cp.empty(1)
    audit_gpu('deepzoom2 eval CUDA context',args.run_dir)
    args.out.mkdir(parents=True,exist_ok=True)
    positions = positions_from(args.positions)
    if args.make_fixture:
        positions = [p for p in positions if p['id']=='m37']
        if not positions:
            raise ValueError('m37 required for engine fixture generation')
    initial_png(positions,args.out)
    rows,fixture_rows,reduction = [],[],None
    previous = {}
    if (args.out/'eval.json').exists():
        old = json.loads((args.out/'eval.json').read_text(encoding='utf-8'))
        previous = {(r['id'],r['eps']):r for r in old['rows']}
    engine_sha = hashlib.sha256(b''.join((F/'deep'/name).read_bytes() for name in
                                ('perturb.py','perturb_kernel.cu','reference.py','limits.py'))).hexdigest()
    start = time.perf_counter()
    for pos in positions:
        for eps in args.depths:
            require_memory()
            n,d = limits.max_iter_for(eps),hp.depth_digits(eps)
            seed = truth.seed_for(pos['id'],eps)
            # These populations are fixed BEFORE the map is computed.
            populations = {'sample625':list(truth.SAMPLE),'uniform':truth.uniform_pixels(seed,args.uniform)}
            m,info = perturb.deep_map(pos['x0'],pos['y0'],eps,max_iter=n)
            audit_gpu(f"deepzoom2 {pos['id']} {eps:g}",args.run_dir)
            populations['band'] = truth.band_pixels(m,seed,args.band)
            digest = hashlib.sha256(m.tobytes()).hexdigest()
            old = previous.get((pos['id'],eps))
            row = dict(id=pos['id'],x0=pos['x0'],y0=pos['y0'],eps=eps,max_iter=n,seed=seed,
                       info=info,structure=structure(m),map_sha256=digest,engine_sha256=engine_sha,
                       band_resampled=old is not None,
                       band_engine_or_map_changed=bool(old and (old['map_sha256']!=digest or old.get('engine_sha256')!=engine_sha)),
                       band_resample_reason='final map recomputed; same deterministic seed reused' if old else None,
                       groups={},n2n={})
            map_png(m,args.out/f"{pos['id']}_{eps:g}.png")
            ladders = {}
            for name,pixels in populations.items():
                points = truth.pixel_points(pos['x0'],pos['y0'],eps,pixels)
                ladders[name] = truth.ladder(points,n,d,args.processes,args.truth_cache)
                row['groups'][name] = population_stats(m,pixels,ladders[name])
                # A reusable detail file retains actual pixels and all ladder tiers.
                write_json(args.out/f"{pos['id']}_{eps:g}_{name}.json",{'pixels':pixels,**ladders[name]})
                if name!='band':
                    twice = truth.ladder(points,2*n,d,args.processes,args.truth_cache)
                    row['n2n'][name] = compare_n2n(ladders[name],twice)
                    write_json(args.out/f"{pos['id']}_{eps:g}_{name}_2N.json",{'pixels':pixels,**twice})
            if not rows:
                elapsed = time.perf_counter()-start
                forecast = elapsed*len(positions)*len(args.depths)
                reduction = {'first_scene_s':elapsed,'projected_total_s':forecast,
                             'first_truth_points_per_second':{k:v['points_per_second'] for k,v in ladders.items()},
                             'reduced':False,'uniform_before':args.uniform,'band_before':args.band}
                if args.reduce_if_slow and forecast>3*3600:
                    args.uniform,args.band = max(1000,min(args.uniform,1000)),max(2500,min(args.band,2500))
                    reduction.update(reduced=True,uniform_after=args.uniform,band_after=args.band,
                                     reason='first-scene projection exceeds 3 hours; reduction applies to subsequent scenes')
            rows.append(row)
            save(args,rows,reduction)
            if args.make_fixture:
                fixture_rows.append(fixture_row(row,populations,ladders))
    if args.make_fixture:
        counterexample = n6_fixture(args)
        data = {'version':2,'generator':'scripts/deepzoom2_eval.py --make-fixture',
                'seconds':time.perf_counter()-start,'positions':fixture_rows,'n6':counterexample}
        write_json(args.make_fixture,data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positions',type=Path)
    parser.add_argument('--depths',nargs='+',type=float,default=DEPTHS)
    parser.add_argument('--uniform',type=int,default=2000)
    parser.add_argument('--band',type=int,default=5000)
    parser.add_argument('--processes',type=int,default=12)
    parser.add_argument('--out',type=Path,default=F/'logs/deepzoom2/eval')
    parser.add_argument('--truth-cache',type=Path,default=F/'logs/deepzoom2/truth_cache')
    parser.add_argument('--reduce-if-slow',action='store_true')
    parser.add_argument('--make-fixture',type=Path)
    add_worker_args(parser,'logs/deepzoom2/eval_guard')
    args = parser.parse_args()
    if args.uniform<0 or args.band<0 or args.processes<1:
        parser.error('sample counts must be nonnegative; processes positive')
    if args.worker:
        run(args)
    else:
        guarded(Path(__file__),args.run_dir)


if __name__=='__main__':
    main()
