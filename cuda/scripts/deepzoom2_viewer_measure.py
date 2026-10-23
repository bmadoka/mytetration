"""Window measurement, or CPU-only summary of an existing threaded run."""
import argparse
import json
from pathlib import Path
import sys

F = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(F), str(F/'scripts'), str(F/'viewer')]
from viewer.deep_view import timing_metrics, state_key
from viewer.deep_state import initial_states, apply_event
from scripts.perturb_tools import write_json


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def job_key(row):
    return state_key((row['x0'], row['y0'], float.fromhex(row['eps_hex'])))


def record_key(record):
    return state_key((*record[:2], float.fromhex(record[3])))


def validate_playback(meta, inputs, frames):
    """Use the original script, never the delivered prefix, as the completion goal."""
    path = Path(meta['script'])
    if not path.is_absolute():
        path = F/path
    script = json.loads(path.read_text(encoding='utf-8'))
    events = script['events']
    if not events or len(inputs) != len(events) or meta['events_delivered'] != len(events):
        raise ValueError('script event count differs from delivered inputs')
    start = meta['t_playback_start']
    state, high = initial_states(script['start'], script.get('start_hp'))
    for k, (expected, actual) in enumerate(zip(events, inputs)):
        if (actual.get('input_index') != k
                or any(name not in actual or actual[name] != value for name, value in expected.items())
                or actual['t_scheduled'] != start+expected['t']):
            raise ValueError(f'script event {k}: number, kind, value or schedule differs')
        if record_key(actual['before_hp']) != state_key(high):
            raise ValueError(f'script event {k}: before state differs')
        state, high, clamped = apply_event(state, high, expected)
        if record_key(actual['after_hp']) != state_key(high) or bool(actual.get('clamped')) != clamped:
            raise ValueError(f'script event {k}: after state differs')
    end = start+script['duration']
    if (script['duration'] < events[-1]['t'] or meta['script_duration'] != script['duration']
            or meta['t_end'] < end or not frames or frames[-1]['t_present'] < end):
        raise ValueError('playback ended before the full script duration')
    target = state_key(high)
    final = next((f for f in frames if f.get('intermediate') is False
                  and f.get('underlying_kind', f.get('present_kind')) == 'deep_final'
                  and f.get('source_stride') == 1 and f['last_input'] == len(events)-1
                  and f['t_present'] >= inputs[-1]['t_processed']
                  and job_key(f['deep']) == target
                  and record_key(f['source_hp']) == target), None)
    if final is None:
        raise ValueError('last script target has no displayed stride 1 final picture')
    return script, final, end


def summarize(directory):
    directory = Path(directory)
    meta = json.loads((directory/'meta.json').read_text(encoding='utf-8'))
    inputs = jsonl(directory/'inputs.jsonl')
    frames = jsonl(directory/'frames.jsonl')
    jobs = jsonl(directory/'deep_jobs.jsonl')
    contamination = json.loads((directory/'contamination.json').read_text(encoding='utf-8'))
    if contamination['verdict']['contaminated']:
        raise ValueError('contaminated run retained but excluded from measurement')
    try:
        if meta['mode'] != 'threaded':
            raise ValueError('measurement requires threaded playback')
        script, last_final, script_end = validate_playback(meta, inputs, frames)
    except (ValueError, KeyError, OSError, TypeError) as error:
        return dict(directory=str(directory), valid=False, status='invalid',
                    measurement_failure=str(error), delivered_input_count=len(inputs))
    finals = [f for f in frames if not f.get('intermediate',True)
              and f.get('underlying_kind',f.get('present_kind')) == 'deep_final']
    # Holds of the same final picture do not move its arrival time to the script end.
    end = max(last_final['t_present'], script_end)
    result = timing_metrics(inputs,frames,inputs[0]['t_scheduled'],end)
    starts = [r for r in jobs if r['event'] == 'start']
    # Count each repeated unit once per extra start; explicit eviction markers
    # also count when the first computation is outside the supplied job history.
    recomputes = {r['id'] for r in jobs if r['event']=='recompute_after_eviction'
                  or (r['event']=='start' and r.get('recompute_after_eviction'))}
    seen = set()
    duplicate_ids = set()
    duplicates = 0
    for row in starts:
        unit = (*job_key(row), row['stride'])
        if unit in seen:
            duplicate_ids.add(row['id'])
            duplicates += 1
        seen.add(unit)
    duplicates += len(recomputes - duplicate_ids)
    stops = []
    for k,ev in enumerate(inputs):
        if k+1 < len(inputs) and inputs[k+1]['t_scheduled']-ev['t_scheduled'] < meta['stop_delay']:
            continue
        key = record_key(ev['after_hp'])
        row = dict(last_input=k,t_last_input_scheduled=ev['t_scheduled'],t_last_input_processed=ev['t_processed'])
        for stride,label in ((8,'coarse'),(1,'final')):
            match = next((r for r in jobs if r['event']=='end' and
                          job_key(r) == key and r['stride']==stride),None)
            shown = next((f for f in frames if match and f.get('source_request_id')==match['id']
                          and f['last_input']==k),None)
            row[label] = None if match is None else dict(request_id=match['id'], t_end=match['t_end'],
                t_received=match['t_received'], t_present=shown['t_present'] if shown else None,
                arrival_after_last_input_ms=1000*(match['t_received']-ev['t_processed']),
                present_after_last_input_ms=1000*(shown['t_present']-ev['t_processed']) if shown else None)
        stops.append(row)
    result.update(directory=str(directory), valid=True, status='measured',
                  script_event_count=len(script['events']), script_duration=script['duration'],
                  final_picture_count=len({f['source_request_id'] for f in finals}),
                  final_after_last_input=last_final is not None, duplicate_jobs=duplicates,
                  recompute_after_eviction=len(recomputes),
                  duplicate_rejections=sum(r['event']=='duplicate_rejected' for r in jobs),
                  range_exit_count=sum(r['event']=='end' and r['info']['out_of_range'] for r in jobs),
                  stops=stops, contamination=contamination,
                  thresholds=dict(input_p95_ms=50,present_p99_ms=50,present_max_ms=200,duplicate_jobs=0),
                  blind_spots='No GPU accuracy, shallow-map regression, visual notice or video judgement.')
    return result


def write_report(directory, result):
    directory = Path(directory)
    write_json(directory/'measure.json',result)
    if result.get('valid') is False:
        (directory/'measure.md').write_text('측정 무효(실패): '+result['measurement_failure']+'\n',encoding='utf-8')
        return
    if 'input_latency_ms' not in result:
        (directory/'measure.md').write_text('3회 오염 기록 보존. 측정에 사용한 깨끗한 회차: 0.\n',encoding='utf-8')
        return
    a,b = result['input_latency_ms'],result['present_interval_ms']
    rows = [('입력 지연 p50 (ms)',a['p50'],'—'),('입력 지연 p95 (ms)',a['p95'],'≤ 50'),
            ('입력 지연 최대 (ms)',a['max'],'—'),('입력 개수',a['n'],'—'),
            ('present 간격 p50 (ms)',b['p50'],'—'),('present 간격 p99 (ms)',b['p99'],'≤ 50'),
            ('present 간격 최대 (ms)',b['max'],'≤ 200'),('중복 계산',result['duplicate_jobs'],'0'),
            ('퇴출 후 재계산',result['recompute_after_eviction'],'0'),
            ('최종 그림 수',result['final_picture_count'],'—'),('범위 이탈',result['range_exit_count'],'—')]
    lines = ['|항목|값|문턱|','|---|---:|---|']+[f'|{label}|{value}|{threshold}|' for label,value,threshold in rows]
    lines += ['',f"측정 구간: {result['measurement_begin']} ~ {result['measurement_end']} (perf_counter).",
              '백분위: numpy linear. 입력 지연은 예정 시각부터 첫 반영 present까지. 간격은 구간을 가로지르는 이웃 present도 포함.',
              f"마지막 입력 뒤 최종 그림 기록: {result['final_after_last_input']}. 미반영 입력: {result['missing_inputs']}.",
              '', '|마지막 입력|거친 그림 도착 / 표시 (ms)|최종 그림 도착 / 표시 (ms)|','|---:|---|---|']
    for stop in result['stops']:
        values = ['없음' if stop[k] is None else f"{stop[k]['arrival_after_last_input_ms']} / {stop[k]['present_after_last_input_ms']}" for k in ('coarse','final')]
        lines.append(f"|{stop['last_input']}|{values[0]}|{values[1]}|")
    lines += ['', '도착은 부모의 결과 수신 시각, 표시는 첫 present. 이미 캐시된 지도는 음수 도착 지연일 수 있다.',
              '못 보는 것: GPU 수치 정확성·얕은 지도 회귀·안내 모양·영상 품질.']
    (directory/'measure.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(directory, script, max_wait=3600, timeout=300, *, readback='none'):
    import cpu_guard
    from viewer import launch
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    try:
        clean = cpu_guard.guarded_run(lambda d: launch.run(script,'threaded',d,readback=readback,max_wait=max_wait,timeout=timeout),directory,attempts=3)
    except cpu_guard.ContaminatedRun:
        records = [str(p) for p in directory.parent.glob(directory.name+'_contaminated_*')]
        write_report(directory,dict(status='not complete (waiting for a clean run)',retained=records,clean_runs=0))
        return
    result = summarize(clean)
    write_report(clean,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=F/'logs/deepzoom2/viewer_measure')
    parser.add_argument('--script',type=Path,default=F/'viewer/scripts/deepzoom2_responsive.json')
    parser.add_argument('--summarize',action='store_true',help='CPU-only existing-log summary')
    parser.add_argument('--max-wait',type=float,default=3600)
    parser.add_argument('--timeout',type=float,default=300)
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.out)
        write_report(args.out,result)
    else:
        result = run(args.out,args.script,args.max_wait,args.timeout)
    if result is not None and result.get('valid') is False:
        sys.exit(1)
