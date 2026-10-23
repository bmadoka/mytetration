"""Evaluate positions × four depths, direct truth, independent references and f64 controls."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from perturb_tools import F, DEPTHS, add_worker_args, audit_gpu, finite_json, guarded, require_memory, write_json
from deep import hp, limits, reference, truth

sys.path.insert(0, str(F/'tests'))
from deepzoom_blocks import compare


def independent_pixel(x,y,eps,n,d,first):
    """A surviving reference other than the first: opposite quadrant first, then the farthest other candidate.

    On threshold boundaries half the view diverges, so the opposite quadrant may hold no
    surviving candidate; fall back to the other surviving candidates ordered by distance from the first pixel.
    """
    p,q = first
    opposite = [(a,b) for a,b in reference.DEFAULT_PIXELS if (a<960)!=(p<960) and (b<540)!=(q<540)]
    others = sorted((c for c in reference.DEFAULT_PIXELS if tuple(c)!=(p,q) and c not in opposite),
                    key=lambda c: -((c[0]-p)**2+(c[1]-q)**2))
    for group in (opposite, others):
        for pixel in group:
            ref, _, _ = reference.choose(x,y,eps,n,d,[pixel])
            if ref.n_ref is None:
                return ref.pixel
    raise RuntimeError('no surviving independent reference among the candidate pixels')


def preview(m, target, path):
    from PIL import Image, ImageDraw
    base = Image.fromarray((m.T[::-1].astype(np.uint8)*255)).convert('RGB')
    base = base.resize((960,540),Image.Resampling.NEAREST)
    samples = Image.new('RGB',(960,540),(40,40,40))
    overlay = base.copy()
    dots, errors = ImageDraw.Draw(samples), ImageDraw.Draw(overlay)
    actual = np.asarray([m[p,q] for p,q in truth.SAMPLE])
    for (p,q), value, mismatch in zip(truth.SAMPLE,target,actual!=target):
        cx,cy = p*959/1919, (1079-q)*539/1079
        dots.ellipse((cx-3,cy-3,cx+3,cy+3),fill='white' if value else 'black')
        if mismatch:
            errors.ellipse((cx-5,cy-5,cx+5,cy+5),outline='red',width=2)
    image = Image.new('RGB',(2880,566),'#303030')
    draw = ImageDraw.Draw(image)
    for i,(panel,label) in enumerate([(base,'Perturb map (white = divergent)'),(samples,'625 direct truth samples'),(overlay,'Sample mismatches (red circles)')]):
        image.paste(panel,(i*960,26))
        draw.text((i*960+8,6),label,fill='white')
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path)


def save(args, rows):
    controls = [r['f64_control'] for r in rows if 'f64_control' in r]
    summary = {'rows':rows,'positions_count':len({(r['x0'],r['y0']) for r in rows}),
               'requirements':{'positions':3,'mixture_each':.05,'sample_mismatch_max':.01,
               'truth_disagreements':0,'target_tiles_min':50,'control_depth':1e-20},
               'control_sample_error_detected':any(c['mismatch_ratio']>.01 for c in controls),
               'control_blocks_detected':any(not c['blocks']['ok'] for c in controls)}
    write_json(args.run_dir/'eval.json',finite_json(summary))
    lines = ['|position|eps|N|truth d1/d2 mismatch|mixture divergent|map mismatch|unique x/y|blocks/targets|range reason|',
             '|---|---:|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        b = r.get('blocks',{})
        lines.append(f"|{r['id']}|{r['eps']:g}|{r['max_iter']}|{r['truth_disagreements']}|{r['diverged_fraction']:.5f}|{r['mismatch_ratio']:.5f}|{r['info']['unique_x']}/{r['info']['unique_y']}|{b.get('block_tiles','-')}/{b.get('target_tiles','-')}|{r['info']['reason']}|")
    lines += ['', '두 번째 지도: 반대 사분면의 비발산 기준 픽셀, 기준 궤도 +40자리.',
              '시간은 contamination.json과 함께 해석한다. 뷰어 표시 fps·드래그·MP4는 이 평가가 보지 못한다.']
    (args.run_dir/'eval_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    require_memory()
    from deep.perturb import deep_map
    import mytet_cuda
    from mytet_cuda import variants
    from mytet_cuda import gpu
    gpu._check_device()
    gpu.cp.empty(1)
    audit_gpu('eval CUDA context', args.run_dir)
    positions = json.loads(args.positions.read_text(encoding='utf-8'))
    if isinstance(positions,dict):
        positions = positions['positions']
    rows = []
    for pos in positions:
        for eps in DEPTHS:
            require_memory()
            x,y = pos['x0'],pos['y0']
            n = limits.max_iter_for(eps)
            d1,d2 = hp.depth_digits(eps)+25,hp.depth_digits(eps)+50
            target1,_ = truth.truth_samples(x,y,eps,n,d1,args.processes,args.truth_cache)
            target2,_ = truth.truth_samples(x,y,eps,n,d2,args.processes,args.truth_cache)
            twice,_ = truth.truth_samples(x,y,eps,2*n,d2,args.processes,args.truth_cache)
            m,info = deep_map(x,y,eps,max_iter=n)
            if info['kernel_passes']:
                audit_gpu(f"{pos['id']} {eps:g} perturb",args.run_dir)
            actual = np.asarray([m[p,q] for p,q in truth.SAMPLE])
            row = {'id':pos['id'],'x0':x,'y0':y,'eps':eps,'max_iter':n,'truth_digits':[d1,d2],
                   'truth_disagreements':int(np.count_nonzero(target1!=target2)),
                   'diverged_fraction':float(target2.mean()), 'nondiverged_fraction':float(1-target2.mean()),
                   'mismatch_count':int(np.count_nonzero(actual!=target2)), 'mismatch_ratio':float(np.mean(actual!=target2)),
                   'limit_changed_fraction':float(np.mean(target2!=twice)), 'limit_position':bool(np.mean(target2!=twice)>.01),
                   'info':info}
            if not info['out_of_range']:
                pixel = independent_pixel(x,y,eps,n,info['ref_digits']+40,info['refs'][0]['pixel'])
                second, info2 = deep_map(x,y,eps,max_iter=n,ref_digits=info['ref_digits']+40,ref_pixel=pixel)
                audit_gpu(f"{pos['id']} {eps:g} independent",args.run_dir)
                row.update(second_info=info2,blocks=compare(m,second))
                row['blocks']['enough_target_tiles'] = row['blocks']['target_tiles']>=50
                # The verdict needs a trusted partner map and >= 50 target tiles.
                row['blocks']['second_out_of_range'] = bool(info2['out_of_range'])
                row['blocks']['condition2_ok'] = bool(row['blocks']['ok'] and row['blocks']['enough_target_tiles']
                                                      and not info2['out_of_range'])
                if eps==1e-20:
                    direct = mytet_cuda.divergence_map('16by9_H',float(x),float(y),eps,1920,n,1e10)
                    audit_gpu(f"{pos['id']} {eps:g} f64 control",args.run_dir)
                    xs,ys,_ = variants.grid('16by9_H',float(x),float(y),eps,1920)
                    sampled = np.asarray([direct[p,q] for p,q in truth.SAMPLE])
                    row['f64_control'] = {'mismatch_ratio':float(np.mean(sampled!=target2)),
                                          'unique_x':len(np.unique(xs)),'unique_y':len(np.unique(ys)),
                                          'blocks':compare(direct,second)}
            filename = f"compare_{pos['id']}_{eps:g}.png"
            if Path(filename).name != filename:
                raise ValueError('position id must be a filename component')
            image = F/'preview_media/perturb'/filename
            preview(m,target2,image)
            row['preview'] = str(image.relative_to(F))
            rows.append(row)
            save(args,rows)
    save(args,rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positions',type=Path,default=F/'logs/perturb/positions/positions.json')
    parser.add_argument('--processes',type=int,default=14)
    parser.add_argument('--truth-cache',type=Path,default=F/'logs/perturb/truth_cache')
    add_worker_args(parser,'logs/perturb/eval')
    args = parser.parse_args()
    if args.worker:
        run(args)
    else:
        guarded(Path(__file__),args.run_dir)


if __name__=='__main__':
    main()
