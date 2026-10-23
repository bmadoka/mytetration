"""Deterministic CPU-only scripts for responsiveness and the 1e-13 crossing."""
import argparse
import json
from pathlib import Path

from deep import hp, limits
from viewer.deep_state import initial_states, apply_event

F = Path(__file__).resolve().parents[1]
DEFAULT_POSITIONS = F/'tests/data/deepzoom2_positions.json'
VIDEO_POSITIONS = F/'tests/data/deepzoom2_positions.json'
VIDEO_DEPTHS = (1e-15, 1e-20, 1e-25, 1e-30, 1e-35, 1e-40)


def exact_final_steps(state):
    """Search positive f64 steps through the actual wheel formula, or fail closed."""
    import struct
    # Positive IEEE doubles have the same ordering as their unsigned bit patterns.
    def value(bits):
        return struct.unpack('>d', struct.pack('>Q', bits))[0]
    lo, hi = 0, struct.unpack('>Q', struct.pack('>d', 1.))[0]
    while lo <= hi:
        mid = (lo+hi)//2
        steps = value(mid)
        eps = hp.apply_wheel(state, steps)[2]
        if eps.hex() == limits.L.hex():
            return steps
        if eps > limits.L:
            lo = mid+1
        else:
            hi = mid-1
    raise ValueError('no final wheel steps reach exactly L')


def position(path=DEFAULT_POSITIONS, ident='m37'):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(rows, dict) and ident in rows:
        row = rows[ident]
    else:
        rows = rows.get('positions', rows) if isinstance(rows,dict) else rows
        row = next(p for p in rows if p['id'] == ident)
    if not all(isinstance(row[k],str) for k in ('x0','y0')):
        raise ValueError('position coordinates must be high precision strings')
    return row


def validate(data, *, crossing=False):
    state, high = initial_states(data['start'], data['start_hp'])
    previous = None
    drags = crosses = 0
    minimum = state[2]
    before = []
    for k, ev in enumerate(data['events']):
        if previous:
            gap = ev['t']-previous['t']
            required = .1 if 'wheel' in (ev['type'],previous['type']) else .016
            if gap < required-1e-9:
                raise ValueError(f'event {k}: interval {gap} < {required}')
            if ev['type'] == previous['type'] == 'drag' and gap < .1 and abs(gap-.016) > 1e-9:
                raise ValueError('continuous drag interval must be 16 ms')
        if ev['type'] == 'drag':
            if sorted((abs(ev['dx']),abs(ev['dy']))) != [0,20]:
                raise ValueError('drag must be 20 render pixels along one axis')
            drags += 1
        elif ev['type'] == 'wheel':
            if abs(ev['steps']) != 1:
                raise ValueError('wheel must be one notch')
        old = high
        state, high, clamped = apply_event(state,high,ev)
        if clamped:
            raise ValueError('script crosses L')
        if (old[2] < limits.DEEP_BELOW) != (high[2] < limits.DEEP_BELOW):
            crosses += 1
            before.append(dict(event=k, before_hp=hp.to_record(old), after_eps_hex=high[2].hex()))
        minimum = min(minimum,state[2])
        previous = ev
    if minimum < limits.L:
        raise ValueError('eps below L')
    if crossing and (drags < 50 or crosses < 6):
        raise ValueError('crossing script needs >=50 drags and >=6 crossings')
    return dict(event_count=len(data['events']), drag_count=drags, crossing_count=crosses,
                min_eps=minimum, crossings=before)


def generate(point):
    def build(eps, crossing):
        events = []
        t = 1.

        def drag(n, axis='dx', sign=None):
            nonlocal t
            # sign None: even blocks return to their original HP centre (round trip).
            # sign +1/-1: net displacement, so stops and crossings happen at NEW states.
            for i in range(n):
                events.append(dict(t=round(t,9),type='drag',dx=0,dy=0))
                events[-1][axis] = (20 if i < n//2 else -20) if sign is None else 20*sign
                t += .016

        def wheels(steps):
            nonlocal t
            t += .1  # no drag in a wheel's 100 ms neighbourhood
            for s in steps:
                events.append(dict(t=round(t,9),type='wheel',steps=s))
                t += .1

        # Every stop and every 1e-13 crossing is at a state not visited before, so the
        # responsiveness run shows coarse -> final refinement on new states (no cache hit) and each crossing
        # checks a different centre after net drags (float/HP drift accumulates between crossings).
        if crossing:
            drag(32,'dx',+1); drag(32,'dy',-1)    # net (+640, -640) px at 3e-13 (shallow): float drift accumulates
            wheels([1]*12)  # 3e-13 -> below 1e-13
            for k in range(3):
                drag(16,'dy',+1)                  # deep side, net +320 px
                wheels([-1]*3)                    # crossing out
                drag(16,'dx',-1 if k%2 else +1)   # shallow side, net +-320 px
                wheels([1]*3)                     # crossing in
            wheels([-1]*3)
            t += 1.5
        else:
            drag(62,'dx',+1)                      # net +1240 px
            t += 1.5                              # stop 1: new centre
            wheels([1]*6)                         # 2e-40 -> 1.06e-40
            t += 1.5                              # stop 2: new depth
            for k in range(4):
                drag(16,'dy',-1)                  # net -320 px each
                wheels([-1,1])
            t += 2.                               # stop 3: new centre
        data = dict(scene='m37', start=[float(point['x0']),float(point['y0']),eps],
                    start_hp=[point['x0'],point['y0']], duration=round(t,9),
                    events=[dict(ev,i=i) for i,ev in enumerate(events)],
                    event_interval_s=.016, wheel_interval_s=.1, drag_pixels=20)
        data.update(validate(data,crossing=crossing))
        return data
    return {'deepzoom2_responsive':build(2e-40,False), 'deepzoom2_crossing':build(3e-13,True)}


def _rel(path):
    path = Path(path).resolve()
    return path.relative_to(F).as_posix() if path.is_relative_to(F) else path.name


def write_scripts(path=DEFAULT_POSITIONS, out=F/'viewer/scripts', ident='m37'):
    files = generate(position(path,ident))
    Path(out).mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        data['scene'] = ident
        data['positions_file'] = _rel(path)
        (Path(out)/(name+'.json')).write_text(json.dumps(data,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    return files


def video_script(point, ident='m37'):
    """HP replayed tour; 16 ms grid, wheels >=100 ms apart, six 1.504 s stops.

    Fractional drags aim at the decimal target. This video-only script intentionally
    differs from the 20 px / one-notch responsiveness and crossing scripts.
    """
    import math
    state = hp.from_floats(0., 0., 5.)
    target = tuple(hp.D(point[k]) for k in ('x0', 'y0'))
    events, corrections, holds = [], [], []
    tick = 62  # initial picture remains on screen for about one second

    def error(s):
        return [hp.CTX.divide(hp.CTX.subtract(s[k], target[k]), spacing)
                for k, spacing in enumerate(hp.spacing(s[2]))]

    def add(gap, **ev):
        nonlocal state, tick
        tick += gap
        ev.update(t=tick*.016, i=len(events))
        events.append(ev)
        state = hp.apply_event(state, ev)

    def center(count):
        for left in range(count, 0, -1):
            dx, dy = error(state)
            add(1, type='drag', dx=float(hp.CTX.divide(dx, hp.D(left))),
                dy=-float(hp.CTX.divide(dy, hp.D(left))))

    center(120)
    zoom_start = len(events)
    max_error = max(map(abs, error(state)))
    for depth in VIDEO_DEPTHS:
        for steps in (1., 1/6):
            while hp.apply_wheel(state, steps)[2] >= depth:
                candidate = hp.apply_wheel(state, steps)
                if max(map(abs, error(candidate))) > hp.D(1):
                    before = hp.to_record(state)
                    center(8)
                    corrections.append(dict(before=before, after=hp.to_record(state),
                                            error_pixels=[str(v) for v in error(state)]))
                add(7, type='wheel', steps=steps)  # 112 ms on the 16 ms grid
                max_error = max(max_error, *map(abs, error(state)))
        if depth == limits.L:
            steps = exact_final_steps(state)
            candidate = hp.apply_wheel(state, steps)
            if max(map(abs, error(candidate))) > hp.D(1):
                before = hp.to_record(state)
                center(8)
                corrections.append(dict(before=before, after=hp.to_record(state),
                                        error_pixels=[str(v) for v in error(state)]))
            add(7, type='wheel', steps=steps)
            max_error = max(max_error, *map(abs, error(state)))
            if state[2].hex() != limits.L.hex():
                raise ValueError('last depth stop must equal L bit for bit')
        if depth != limits.L and not depth <= state[2] < 1.02*depth:
            raise ValueError('depth stop does not lie in [depth, 1.02 depth)')
        holds.append(dict(depth=depth, eps=state[2], eps_hex=state[2].hex(),
                          event=events[-1]['i'], start=events[-1]['t'],
                          end=(tick+94)*.016, hp=hp.to_record(state)))
        tick += 94  # 1.504 s: the first multiple of 16 ms >=1.5 s
    data = dict(scene=ident, start=[0., 0., 5.], duration=tick*.016, events=events,
                position=dict(point, id=ident), holds=holds, corrections=corrections,
                zoom_start_event=zoom_start, max_center_error_pixels=str(max_error),
                end_hp=hp.to_record(state), event_interval_s=.016, wheel_interval_s=.112,
                format_exceptions=[
                    'Video only, not the responsiveness/crossing scripts: fractional residual drags instead of 20 px.',
                    'Wheels use 112 ms (7 x 16 ms), the smallest 16 ms grid interval >=100 ms.',
                    'Wheels use one notch, then 1/6 near depth stops; the last residual notch reaches exactly L (float bits).',
                    'Stops use 1.504 s (94 x 16 ms), the smallest grid interval >=1.5 s.'])
    if not math.isfinite(float(max_error)) or max_error > hp.D(3):
        raise ValueError('tour loses its HP target by more than three pixels')
    return data


def write_video_script(path=VIDEO_POSITIONS, out=F/'viewer/scripts/deepzoom2_video.json', ident='m37'):
    data = video_script(position(path, ident), ident)
    data['positions_file'] = _rel(path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positions',type=Path)
    parser.add_argument('--position',default='m37')
    parser.add_argument('--out',type=Path,default=F/'viewer/scripts')
    parser.add_argument('--video',action='store_true',help='write deepzoom2_video.json only')
    args = parser.parse_args()
    if args.video:
        write_video_script(args.positions or VIDEO_POSITIONS,args.out/'deepzoom2_video.json',args.position)
    else:
        write_scripts(args.positions or DEFAULT_POSITIONS,args.out,args.position)
