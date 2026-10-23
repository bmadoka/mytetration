"""Crossing analysis. --frames-only is CPU-only; map comparison needs the GPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

F = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(F),str(F/'scripts'),str(F/'viewer')]
from deep import hp
from viewer.deep_view import warp_map
from scripts.deepzoom2_viewer_measure import jsonl
from scripts.perturb_tools import write_json


def boundary_pairs(m):
    return int(np.count_nonzero(m[1:] != m[:-1]) + np.count_nonzero(m[:,1:] != m[:,:-1]))


def scan_offsets(shallow, deep, center, *, radius=5):
    es,ed = 1e-13*(1+1e-4),1e-13*(1-1e-4)
    source = hp.from_strings(*center,es)
    target = hp.from_strings(*center,ed)
    aligned, transform = warp_map(shallow,target,source)
    rows = []
    nx,ny = deep.shape
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            xa,xb = max(0,dx), min(nx,nx+dx)
            ya,yb = max(0,dy), min(ny,ny+dy)
            a,b = aligned[xa:xb,ya:yb],deep[xa-dx:xb-dx,ya-dy:yb-dy]
            count = int(np.count_nonzero(a==b))
            rows.append(dict(offset=[dx,dy], matching=count, overlap=a.size, agreement=count/a.size))
    best = rows[0]
    for row in rows[1:]:
        if row['matching']*best['overlap'] > best['matching']*row['overlap']:
            best=row
    ties = [r['offset'] for r in rows if r['matching']*best['overlap']==best['matching']*r['overlap']]
    boundaries = [boundary_pairs(shallow),boundary_pairs(deep)]
    return dict(max_offset=best['offset'],max_agreement=best['agreement'],ties=ties,
                zero_in_ties=[0,0] in ties,boundary_pairs=boundaries,
                counted=min(boundaries)>=1000,scan=rows,scale_transform=transform,
                alignment='Resample shallow onto deep grid: same Decimal HP center, eps ratio, nearest floor(q+0.5), edge clamp. Float-center rounding is intentionally not compensated.',
                offset_definition='aligned[x+dx,y+dy] compared with deep[x,y] on overlapping pixels')


def frame_summary(frames):
    rows=[]
    for f in frames:
        diff,limit = f['center_diff_px'],f['f64_round_limit_px']
        rows.append(dict(seq=f['seq'],center_diff_px=diff,f64_round_limit_px=limit,
                         thresholds_px=[max(1,v) for v in limit], hp_center=f['hp_center'],
                         display_eps=f['display_params'][2]))
    return dict(frames=rows,frame_count=len(rows),
                max_abs_center_diff_px=[max((abs(r['center_diff_px'][axis]) for r in rows),default=0.) for axis in (0,1)],
                excess_axis_count=sum(abs(d)>max(1,b) for r in rows for d,b in zip(r['center_diff_px'],r['f64_round_limit_px'])))


def crossing_states(directory,frames):
    path=Path(directory)/'inputs.jsonl'
    if path.exists():
        return [dict(event=k,before_hp=r['before_hp']) for k,r in enumerate(jsonl(path)) if r['crossed_deep']]
    rows=[]
    for previous,current in zip(frames,frames[1:]):
        if (previous['display_params'][2]<1e-13) != (current['display_params'][2]<1e-13):
            rows.append(dict(event=current['last_input'],before_hp=previous['hp_center']))
    return rows


def analyze(directory,out,*,maps=False):
    frames=jsonl(Path(directory)/'frames.jsonl')
    result=frame_summary(frames)
    inputs=jsonl(Path(directory)/'inputs.jsonl') if (Path(directory)/'inputs.jsonl').exists() else []
    result['resync_count']=sum(bool(r.get('resync')) for r in inputs)
    crosses=crossing_states(directory,frames)
    result.update(crossings=[],recorded_crossings=len(crosses),map_comparison_computed=maps)
    if maps:
        from scripts.perturb_tools import require_memory,audit_gpu
        require_memory()
        from mytet_cuda import gpu
        gpu.device_info()
        gpu.cp.empty(1)
        result['gpu_process']=audit_gpu('crossing context',out)
        from viewer import fast
        from deep import perturb
        for row in crosses:
            center=row['before_hp'][:2]
            shallow=fast.precise_map(float(center[0]),float(center[1]),1e-13*(1+1e-4))
            deep,info=perturb.deep_map(*center,1e-13*(1-1e-4))
            comparison=scan_offsets(shallow,deep,center)
            if info['out_of_range']:
                comparison['counted']=False
            comparison.update(row,info=info)
            result['crossings'].append(comparison)
    result['counted_crossings']=sum(r['counted'] for r in result['crossings'])
    result['blind_spots']='Frame summaries do not measure visible jumps. Offset agreement does not establish deep numerical truth; near-monochrome / range-exit crossings are excluded.'
    write_json(Path(out)/'crossing.json',result)
    lines=['|항목|값|','|---|---:|',f"|재동기 횟수|{result['resync_count']}|",'',
           f"프레임 수: {result['frame_count']}; 축별 |중심 차| 최대(px): {result['max_abs_center_diff_px']}; 문턱 초과 축 수: {result['excess_axis_count']}.",
           f"기록된 넘기: {len(crosses)}; 지도 검사: {maps}; 경계 쌍 각각 ≥1000이며 범위 이탈 없는 넘기: {result['counted_crossings']} (요구 수 6).",'',
           '|입력|최대 오프셋|동률 목록|(0,0) 포함|경계 쌍(얕은/깊은)|계수 대상|','|---:|---|---|---|---|---|']
    for r in result['crossings']:
        lines.append(f"|{r['event']}|{r['max_offset']}|{r['ties']}|{r['zero_in_ties']}|{r['boundary_pairs']}|{r['counted']}|")
    lines+=['','축척 맞춤: 동일한 Decimal 중심으로 얕은 그림을 깊은 eps 격자에 최근접(floor(q+0.5)) 재표본화하고 범위 밖은 끝 화소로 채운다. float 중심 반올림 차는 보정하지 않는다. 그 뒤 ±5 오프셋의 겹친 영역만 비교한다.',
            '문턱: 축별 |차| ≤ max(1, 0.5 ulp / Decimal 화소 간격). 최대 일치 오프셋 (0,0)±1, 동률이면 (0,0) 포함.',
            '못 보는 것: 중심 기록만으로 실제 창의 튐을 보지 못한다. 그림 간 일치율은 고정밀 정답 대조를 대신하지 않는다.']
    (Path(out)/'crossing.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=F/'logs/deepzoom2/crossing')
    parser.add_argument('--run',action='store_true',help='first run the protected window script')
    parser.add_argument('--frames-only',action='store_true',help='no GPU map comparison')
    parser.add_argument('--max-wait',type=float,default=3600)
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.worker:
        analyze(args.directory,args.run_dir,maps=True)
    else:
        if args.run:
            from viewer import launch
            import cpu_guard
            cpu_guard.guarded_run(lambda d: launch.run(F/'viewer/scripts/deepzoom2_crossing.json','threaded',d,
                                  max_wait=args.max_wait,timeout=args.timeout),args.directory,attempts=3)
        if args.frames_only:
            analyze(args.directory,args.directory)
        else:
            from scripts.perturb_tools import guarded
            import shutil
            clean=guarded(Path(__file__),args.directory/'crossing_gpu')
            for name in ('crossing.json','crossing.md'):
                shutil.copyfile(Path(clean)/name,args.directory/name)
