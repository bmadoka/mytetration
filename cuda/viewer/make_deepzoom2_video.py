"""Protected window recording, CPU-only encoding and evaluation-PNG collection.

python -m viewer.make_deepzoom2_video run
python -m viewer.make_deepzoom2_video video logs/deepzoom2/video
python -m viewer.make_deepzoom2_video scenes logs/deepzoom2/eval
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
sys.path.insert(0, str(F/'scripts'))
from deep.limits import L
from viewer.deepzoom2_scripts import VIDEO_DEPTHS, VIDEO_POSITIONS, write_video_script
from viewer.deep_view import state_key
from viewer.make_realtime_video import (timeline, compose_frame, choose_crf,
                                        FFMPEG, FFPROBE, FONT, W, H)

OUT = F/'preview_media/deepzoom2'
SEND_BYTES = 48_000_000


def path_label(frame):
    kind = frame.get('underlying_kind', frame.get('present_kind', ''))
    if frame.get('compute_path') == 'perturb':
        return {'deep_warped': '깊은 옮김', 'deep_coarse': '깊은 거친',
                'deep_final': '깊은 최종', 'deep_notice': '깊은 안내'}.get(kind, '깊은 옮김')
    if frame.get('compute_path') == 'warped' or kind == 'warped':
        return '얕은 옮김'
    if frame.get('compute_path') == 'double' or frame.get('kind') != 'fast':
        return '얕은 정밀'
    return '얕은 빠른'


def compose(bits, frame, t, present, inputs, begin, fonts):
    from PIL import Image, ImageDraw
    rgb = compose_frame(bits, frame, t, present, inputs, begin, fonts)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, W-11, 145), fill='black')
    font, bold = fonts
    draw.text((20, 14), f'실시간 1×   {t-begin:7.2f} s', font=font, fill='white')
    draw.text((20, 42), f"eps {frame['params'][2]:.3e}", font=font, fill='white')
    draw.text((20, 70), path_label(frame), font=bold, fill='white')
    draw.text((20, 100), frame.get('notice_text', ''), font=font, fill='white')
    return np.asarray(image)


def load_recording(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir/'meta.json').read_text(encoding='utf-8'))
    frames = [json.loads(s) for s in (run_dir/'frames.jsonl').read_text(encoding='utf-8').splitlines() if s]
    bits = np.load(run_dir/'window_frames_bits.npy', mmap_mode='r', allow_pickle=False)
    if (not frames or len(bits) != len(frames) or bits.dtype != np.uint8
            or bits.shape != (len(frames), (W*H+7)//8) or meta['recorded_non_binary_pixels'] != 0
            or meta.get('recorded_frames') != len(frames)):
        raise ValueError('incomplete/nonbinary window recording')
    if meta['mode'] not in ('threaded', 'timed'):
        raise ValueError('recording needs threaded or timed playback')
    contamination = json.loads((run_dir/'contamination.json').read_text(encoding='utf-8'))
    if contamination['verdict']['contaminated'] is not False:
        raise ValueError('contaminated recording cannot supply the video')
    if frames[0]['params'] != [0., 0., 5.]:
        raise ValueError('tour must start at (0,0,5)')
    if float(frames[-1]['params'][2]).hex() != L.hex():
        raise ValueError('tour must finish in exactly L (float bits)')
    if (frames[-1].get('underlying_kind', frames[-1].get('present_kind')) != 'deep_final'
            or frames[-1].get('intermediate') is not False):
        raise ValueError('tour must finish with the actual final picture')
    present = np.asarray([f['t_present'] for f in frames], dtype=float)
    if not np.isfinite(present).all() or (np.diff(present) <= 0).any():
        raise ValueError('present times must be finite and strictly increasing')
    fps = float(meta['monitor_refresh_hz'])
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(meta['t_end']) or meta['t_end'] < present[-1]:
        raise ValueError('invalid recording clock')
    script_path = Path(meta['script'])
    if not script_path.is_absolute():
        script_path = F/script_path
    script = json.loads(script_path.read_text(encoding='utf-8'))
    if [h['depth'] for h in script['holds']] != list(VIDEO_DEPTHS):
        raise ValueError('recording script must contain all six depth stops')
    stops = []
    for hold in script['holds']:
        begin = meta['t_playback_start']+hold['start']
        end = meta['t_playback_start']+hold['end']
        if hold['end']-hold['start'] < 1.5-1e-9:
            raise ValueError('depth stop shorter than 1.5 s')
        actual = [f for f in frames if begin <= f['t_present'] <= end
                  and f.get('underlying_kind', f.get('present_kind')) == 'deep_final'
                  and f.get('intermediate') is False
                  and state_key((f['deep']['x0'], f['deep']['y0'], f['params'][2]))
                      == state_key((*hold['hp'][:2], float.fromhex(hold['eps_hex'])))]
        if not actual:
            raise ValueError(f"no final picture during depth stop {hold['depth']:g}")
        stops.append(dict(depth=hold['depth'], first_final_present=actual[0]['t_present'],
                          final_present_count=len(actual), scheduled_start=begin, scheduled_end=end))
    return meta, frames, bits, contamination, stops


def video(run_dir, out=OUT):
    from PIL import ImageFont
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destinations = {k: out/f'deepzoom2_zoom_{k}.mp4' for k in ('archive', 'send')}
    if (out/'video.json').exists() or any(p.exists() for p in destinations.values()):
        raise FileExistsError('video outputs already exist; retain them and choose --out')
    meta, frames, bits, contamination, stops = load_recording(run_dir)
    present = [f['t_present'] for f in frames]
    fps = float(meta['monitor_refresh_hz'])
    begin, end = present[0], max(meta['t_end'], present[-1]+1.5/fps)
    times, indices = timeline(present, begin, end, fps=fps, phase=.5)
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(frames):
        raise ValueError('invalid recording timeline')
    inputs = sorted(meta['input_t'].values())
    fonts = (ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT.replace('malgun', 'malgunbd'), 18))
    encoded = {}

    def encode(crf):
        if crf in encoded:
            return encoded[crf]['size_bytes']
        path = out/f'deepzoom2_encode_crf{crf}.mp4'
        if path.exists():
            raise FileExistsError(path)
        proc = subprocess.Popen([FFMPEG, '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                 '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                                 '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                                 '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        try:
            for t, i in zip(times, indices):
                proc.stdin.write(compose(bits[i], frames[i], t, present, inputs, begin, fonts).tobytes())
            proc.stdin.close()
            code = proc.wait()
        except BaseException:
            proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            raise
        if code:
            raise RuntimeError(f'ffmpeg exit {code}')
        encoded[crf] = dict(file=str(path), size_bytes=path.stat().st_size, crf=crf)
        return encoded[crf]['size_bytes']

    encode(28)
    shutil.copyfile(encoded[28]['file'], destinations['archive'])
    chosen, attempts = choose_crf(encode, max_bytes=SEND_BYTES, max_crf=51)
    result = dict(run=str(run_dir), outputs=dict(archive=dict(encoded[28], file=str(destinations['archive']))),
                  send_attempts=attempts, presented_frames=len(frames), video_frames=len(times), fps=fps,
                  duration_s=len(times)/fps, recorded_duration_s=meta['t_end']-begin,
                  appended_hold_s=end-meta['t_end'], start=frames[0]['params'], end=frames[-1]['params'],
                  start_hp=meta['start_hp'], end_hp=meta['end_hp'], depth_stops=stops,
                  contamination=contamination, encoding_attempts=encoded,
                  blind_spots='Captured window pixels do not establish numerical accuracy, structure, or latency thresholds.')
    if chosen is not None:
        shutil.copyfile(encoded[chosen]['file'], destinations['send'])
        result['outputs']['send'] = dict(encoded[chosen], file=str(destinations['send']))
    else:
        result['error'] = 'No CRF 28..51 fits 48,000,000 B'
    try:
        for record in result['outputs'].values():
            probe = subprocess.run([FFPROBE, '-v', 'error', '-count_frames', '-show_entries',
                'stream=codec_type,codec_name,width,height,nb_read_frames:format=duration,size',
                '-of', 'json', record['file']], capture_output=True, text=True, check=True)
            record['ffprobe'] = data = json.loads(probe.stdout)
            stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
            if (int(stream['nb_read_frames']) != len(times) or (stream['width'], stream['height']) != (W, H)
                    or abs(float(data['format']['duration'])-len(times)/fps) > max(.01, 1/fps)):
                raise ValueError('encoded frame count, size or duration differs from timeline')
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'video.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    if chosen is None:
        raise RuntimeError(result['error'])
    return result


def collect_scenes(eval_dir, out=OUT, positions=None):
    """Copy the evaluator's real PNGs; require every position x (start + six depths)."""
    from PIL import Image, ImageDraw
    eval_dir, out = Path(eval_dir), Path(out)
    data = json.loads((eval_dir/'eval.json').read_text(encoding='utf-8'))
    available = list(dict.fromkeys(r['id'] for r in data['rows']))
    names = available if positions is None else list(positions)
    if len(set(names)) != len(names) or any(n not in available for n in names):
        raise ValueError('requested scene positions must be unique and present in evaluation')
    if not names or any(Path(n).name != n or any(c in n for c in '/\\:') for n in names):
        raise ValueError('invalid evaluation position names')
    keys = [(r['id'], r['eps']) for r in data['rows']]
    if any(keys.count((n, e)) != 1 for n in names for e in VIDEO_DEPTHS):
        raise ValueError('evaluation needs one scene per position and depth')
    sources = [(n, label, eval_dir/('initial.png' if label == 'start' else f'{n}_{label}.png'))
               for n in names for label in ('start', *(f'{e:g}' for e in VIDEO_DEPTHS))]
    dest = out/'scenes'
    sheet_path = out/'scenes_contact_sheet.png'
    if dest.exists() or sheet_path.exists():
        raise FileExistsError('scene collection exists; retain it and choose --out')
    for _, _, p in sources:
        with Image.open(p) as im:
            im.verify()
    dest.mkdir(parents=True)
    cell_w, cell_h = 240, 162
    sheet = Image.new('RGB', (7*cell_w, len(names)*cell_h), 'black')
    draw = ImageDraw.Draw(sheet)
    records = []
    for i, (name, label, source) in enumerate(sources):
        target = dest/f'{name}_{label}.png'
        shutil.copyfile(source, target)
        with Image.open(target) as im:
            thumb = im.convert('RGB')
            thumb.thumbnail((cell_w, 135))
            sheet.paste(thumb, ((i%7)*cell_w, (i//7)*cell_h+24))
        draw.text(((i%7)*cell_w+4, (i//7)*cell_h+4), f'{name}  {label}', fill='white')
        records.append(dict(source=str(source), file=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    sheet.save(sheet_path)
    report = dict(positions=names, evaluation_positions=available, columns=['start', *VIDEO_DEPTHS], scenes=records,
                  contact_sheet=str(sheet_path), blind_spots='PNG copies and thumbnails do not validate evaluator accuracy.')
    (out/'scenes.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def record(directory, *, positions=VIDEO_POSITIONS, ident='m37', max_wait=3600, timeout=900):
    from viewer import launch
    from cpu_guard import guarded_run
    script = F/'viewer/scripts/deepzoom2_video.json'
    write_video_script(positions, script, ident)
    return guarded_run(lambda d: launch.run(script, 'threaded', d, readback='record',
                                            max_wait=max_wait, timeout=timeout), directory, attempts=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('script', 'run', 'video', 'scenes'))
    parser.add_argument('directory', nargs='?', type=Path, default=F/'logs/deepzoom2/video')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--positions', type=Path, default=VIDEO_POSITIONS)
    parser.add_argument('--position', default='m37')
    parser.add_argument('--scene-positions', nargs='+', help='collect only these completed evaluation positions')
    parser.add_argument('--max-wait', type=float, default=3600)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    if args.action == 'script':
        write_video_script(args.positions, ident=args.position)
    elif args.action == 'scenes':
        collect_scenes(args.directory, args.out, args.scene_positions)
    else:
        directory = record(args.directory, positions=args.positions, ident=args.position,
                           max_wait=args.max_wait, timeout=args.timeout) if args.action == 'run' else args.directory
        video(directory, args.out)


if __name__ == '__main__':
    main()
