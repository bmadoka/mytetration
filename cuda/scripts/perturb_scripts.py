"""CPU-only deterministic perturbation scripts, with a fully HP-replayed video tour."""
import json
import sys
from pathlib import Path

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
from deep import hp, limits
from viewer.deep_state import initial_states, apply_event

DEFAULT_POINTS = [
    {'id': 'e1', 'x0': '-0.465275136210219115944123957708250601447510212439516253360016', 'y0': '0.918010934455088400000000000000000000000000000000000000000000'},
    {'id': 'e4', 'x0': '-0.464957088724448366281188803875968093915949199580937761985176', 'y0': '0.918043077312231200000000000000000000000000000000000000000000'},
    {'id': 'e8', 'x0': '-0.493676369279863752470681196304794353861836635494616081966063', 'y0': '0.923693660377358400000000000000000000000000000000000000000000'},
    {'id': 'e9', 'x0': '-0.962017845357889110296869514415342063940913344101010250547159', 'y0': '0.782429245283018800000000000000000000000000000000000000000000'},
]


def positions(path=None):
    path = Path(path) if path else F/'logs/perturb/positions/positions.json'
    rows = json.loads(path.read_text(encoding='utf-8')) if path.exists() else DEFAULT_POINTS
    if isinstance(rows, dict):
        rows = rows['positions']
    if not rows:
        raise ValueError('positions file is empty')
    if any(not isinstance(p[k], str) for p in rows for k in ('id', 'x0', 'y0')):
        raise ValueError('position id/x0/y0 must be strings')
    return rows


def replay(data):
    state, high = initial_states(data['start'], data.get('start_hp'))
    clamped = []
    for ev in data['events']:
        state, high, clamp = apply_event(state, high, ev)
        if clamp:
            clamped.append(ev['i'])
    return state, high, clamped


def video_script(point):
    state = hp.from_floats(0., 0., 5.)
    target = hp.D(point['x0']), hp.D(point['y0'])
    events, corrections = [], []
    t = 3.

    def add(at, **ev):
        nonlocal state
        ev.update(t=at, i=len(events))
        events.append(ev)
        state = hp.apply_event(state, ev)

    def drag_to(at, seconds, dest, count):
        # Last residual is computed again to avoid summation drift; fractional pixels are intentional.
        for i in range(count):
            sx, sy = hp.spacing(state[2])
            remain = hp.D(count-i)
            dx = hp.CTX.divide(hp.CTX.subtract(state[0], dest[0]), hp.CTX.multiply(sx, remain))
            dy = hp.CTX.divide(hp.CTX.subtract(dest[1], state[1]), hp.CTX.multiply(sy, remain))
            add(at+seconds*(i+1)/count, type='drag', dx=float(dx), dy=float(dy))

    drag_to(1., 2., tuple(hp.D(float(v)) for v in target), 120)
    pending = [1e-14, 1e-29]
    while True:
        next_state = hp.apply_wheel(state, 1/6)
        if next_state[2] < limits.L:
            break
        t += 1/60
        add(t, type='wheel', steps=1/6)
        if pending and state[2] <= pending[0]:
            threshold = pending.pop(0)
            before = hp.to_record(state)
            drag_to(t, .5, target, 30)
            t += .5
            corrections.append({'threshold': threshold, 'before': before, 'after': hp.to_record(state)})
    data = {'scene': point['id'], 'start': [0., 0., 5.], 'duration': t+1., 'events': events}
    float_end, high, clamps = replay(data)
    assert high == state and not clamps
    dx, dy = hp.CTX.subtract(high[0], target[0]), hp.CTX.subtract(high[1], target[1])
    sx, sy = hp.spacing(high[2])
    diag = {'position': point, 'end_center': [str(v) for v in high[:2]], 'end_state': list(float_end),
            'eps_hex': high[2].hex(), 'delta': [str(dx), str(dy)],
            'delta_over_eps': [str(hp.CTX.divide(v, hp.D(high[2]))) for v in (dx,dy)],
            'delta_pixels': [str(hp.CTX.divide(dx,sx)), str(hp.CTX.divide(dy,sy))],
            'corrections': corrections, 'events': len(events), 'duration': data['duration']}
    return data, diag


def generate(point):
    def script(eps, duration, events):
        return {'scene': point['id'], 'start': [float(point['x0']),float(point['y0']),eps],
                'start_hp': [point['x0'],point['y0']], 'duration': duration,
                'events': [dict(ev,i=i) for i,ev in enumerate(events)]}
    files = {
        'perturb_cross': script(4e-13, 5., [dict(t=(i+1)/60,type='wheel',steps=1/6) for i in range(240)]),
        'perturb_limit': script(1.1e-40, 3., [dict(t=1+i*.5,type='wheel',steps=1.) for i in range(3)]),
        'perturb_drag': script(1e-40, 2.5, [dict(t=1.,type='drag',dx=-100.,dy=0.)]),
    }
    for d in (15,20,30,40):
        events = [dict(t=1+(i+1)/30,type='wheel',steps=-1/6) for i in range(60)]
        events += [dict(t=3+(i+1)/30,type='wheel',steps=1/6) for i in range(60)]
        files[f'perturb_depth_{d}'] = script(10.**-d, 6., events)
    files['perturb_video'], diag = video_script(point)
    return files, diag


def write_scripts(point=None):
    files, diag = generate(point or positions()[0])
    destination = F/'viewer/scripts'
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (destination/f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    path = F/'logs/perturb/video_script.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diag, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return files, diag


if __name__ == '__main__':
    write_scripts()
