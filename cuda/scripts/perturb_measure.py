"""Four-depth window measurement, using the unchanged CPU guard."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
sys.path.insert(0, str(F/'viewer'))
from perturb_tools import write_json


def summarize(directory):
    directory = Path(directory)
    frames = [json.loads(s) for s in (directory/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    deep = [f for f in frames if f['compute_path'] == 'perturb']
    ms = [f['compute_ms'] for f in deep]
    # The exact start depth separately (initial + final precise frames sit at the start eps),
    # and the eps range actually covered by the frames that the overall statistics mix.
    meta_path = directory/'meta.json'
    start = (json.loads(meta_path.read_text(encoding='utf-8'))['start'] if meta_path.exists()
             else frames[0]['display_params'])  # the initial frame is drawn at the start state
    start_hex = float(start[2]).hex()
    at_start = [f['compute_ms'] for f in deep if f['display_params_hex'][2] == start_hex]
    eps_values = [f['display_params'][2] for f in deep]
    moving = [f['t_present'] for f in deep if f['kind'] == 'fast' and f['last_input'] >= 0]
    intervals = np.diff(moving)
    return {'directory': str(directory), 'deep_frames': len(deep),
            'start_eps': float(start[2]), 'start_depth_frames': len(at_start),
            'start_depth_compute_ms': {'median': float(np.median(at_start)) if at_start else None,
                                       'values': at_start},
            'eps_range': [min(eps_values), max(eps_values)] if eps_values else None,
            'compute_ms': {'median': float(np.median(ms)) if ms else None,
                           'p95': float(np.percentile(ms,95)) if ms else None,
                           'max': max(ms) if ms else None},
            'moving_present_intervals_s': intervals.tolist(),
            'moving_fps': float(1/np.mean(intervals)) if len(intervals) else None,
            'moving_frame_p95_ms': float(1000*np.percentile(intervals,95)) if len(intervals) else None,
            'range_reasons': sorted({f['deep']['reason'] for f in deep if f['deep']['out_of_range']}),
            'contamination': json.loads((directory/'contamination.json').read_text(encoding='utf-8'))['verdict']}


def run(out):
    import launch
    import cpu_guard
    from perturb_scripts import write_scripts
    write_scripts()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in (15,20,30,40):
        directory = out/f'depth_{depth}'
        if directory.exists() or list(out.glob(f'depth_{depth}_contaminated_*')):
            raise FileExistsError(f'existing measurement: {directory}')
        exhausted = False
        try:
            cpu_guard.guarded_run(lambda d: launch.run(F/f'viewer/scripts/perturb_depth_{depth}.json',
                                  'timed', d, readback='none'), directory)
        except cpu_guard.ContaminatedRun:
            exhausted = True
        paths = sorted(out.glob(f'depth_{depth}_contaminated_*'))
        if directory.exists():
            paths.append(directory)
        for attempt, path in enumerate(paths, 1):
            rows.append(dict(summarize(path), depth=depth, attempt=attempt, all_attempts_contaminated=exhausted))
        write_json(out/'measure.json', {'runs': rows, 'status': '완료 아님(측정 대기)' if any(r['all_attempts_contaminated'] for r in rows) else 'measured',
                                       'fps_definition': '1 / mean consecutive moving fast deep present interval',
                                       'blind_spots': 'No accuracy, shallow regression, or video judgement.'})
        lines = ['|depth|attempt|start-eps compute median ms (n)|all-frames median ms|p95 ms|max ms|eps range|moving fps|contamination verdict|',
                 '|---:|---:|---:|---:|---:|---:|---|---:|---|']
        for r in rows:
            c = r['compute_ms']
            sd = r['start_depth_compute_ms']
            lines.append(f"|{r['depth']}|{r['attempt']}|{sd['median']} ({r['start_depth_frames']})|{c['median']}|{c['p95']}|{c['max']}|{r['eps_range']}|{r['moving_fps']}|{json.dumps(r['contamination'])}|")
        lines += ['', 'fps = 움직이는 fast 섭동 프레임의 연속 present 간격 평균의 역수. 목표 문턱 없음.',
                  '오염 회차도 보존한다. 정확성·얕은 구간 회귀·영상은 이 측정이 보지 못한다.']
        (out/'measure.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=F/'logs/perturb/measure')
    run(parser.parse_args().out)
