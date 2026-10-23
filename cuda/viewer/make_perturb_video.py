"""Record the deep tour (opens a window) or encode captured window bits without CUDA.

python viewer/make_perturb_video.py run
python viewer/make_perturb_video.py video logs/perturb/video
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
sys.path.insert(0, str(F/'scripts'))
from viewer.make_realtime_video import timeline, compose_frame, choose_crf, FFMPEG, FFPROBE, FONT, W, H
from deep.limits import L

OUT = F/'preview_media/perturb'
SEND_BYTES = 48_000_000


def path_label(frame):
    path = frame['compute_path']
    if path == 'perturb':
        return '섭동'
    if path == 'double':
        return '정밀 double'
    if path == 'warped':
        return '옮김 ' + frame.get('precision', 'f64')
    return '빠른 ' + path


def compose(bits, frame, t, present, inputs, begin, fonts):
    from PIL import Image, ImageDraw
    # Reuse the established binary readback conversion and composition.
    rgb = compose_frame(bits, frame, t, present, inputs, begin, fonts)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10,10,949,145), fill='black')
    font, bold = fonts
    draw.text((20,14), f'실시간 1×   {t-begin:6.2f} s', font=font, fill='white')
    draw.text((20,42), f"eps {frame['params'][2]:.3e}", font=font, fill='white')
    draw.text((20,70), path_label(frame), font=bold, fill='white')
    draw.text((20,100), frame.get('notice_text', ''), font=font, fill='white')
    return np.asarray(image)


def video(run_dir, out=OUT):
    from PIL import ImageFont
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destinations = {k: out/f'perturb_zoom_{k}.mp4' for k in ('archive','send')}
    if any(p.exists() for p in destinations.values()):
        raise FileExistsError('archive/send already exists; retain it and choose --out')
    meta = json.loads((run_dir/'meta.json').read_text(encoding='utf-8'))
    frames = [json.loads(s) for s in (run_dir/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    bits = np.load(run_dir/'window_frames_bits.npy', mmap_mode='r')
    if not frames or len(bits) != len(frames) or meta['recorded_non_binary_pixels'] != 0:
        raise ValueError('incomplete/nonbinary window recording')
    if frames[0]['params'] != [0.,0.,5.] or frames[-1]['compute_path'] != 'perturb':
        raise ValueError('tour must start at (0,0,5) and finish in perturbation')
    if not L <= frames[-1]['params'][2] < 1.02*L:
        raise ValueError('tour must finish near L without crossing it')
    present = [f['t_present'] for f in frames]
    fps = float(meta['monitor_refresh_hz'])
    begin, end = present[0], max(meta['t_end'],present[-1]+1.5/fps)
    times, indices = timeline(present,begin,end,fps=fps,phase=.5)
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(frames):
        raise ValueError('invalid recording timeline')
    inputs = sorted(meta['input_t'].values())
    fonts = (ImageFont.truetype(FONT,17),ImageFont.truetype(FONT.replace('malgun','malgunbd'),18))
    encoded = {}

    def encode(crf):
        if crf in encoded:
            return encoded[crf]['size_bytes']
        path = out/f'perturb_encode_crf{crf}.mp4'
        if path.exists():
            raise FileExistsError(path)
        proc = subprocess.Popen([FFMPEG,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
                                 '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','slow','-crf',str(crf),
                                 '-pix_fmt','yuv420p','-movflags','+faststart',str(path)], stdin=subprocess.PIPE)
        try:
            for t, i in zip(times,indices):
                proc.stdin.write(compose(bits[i],frames[i],t,present,inputs,begin,fonts).tobytes())
        finally:
            proc.stdin.close()
            code = proc.wait()
        if code:
            raise RuntimeError(f'ffmpeg exit {code}')
        encoded[crf] = {'file': str(path), 'size_bytes': path.stat().st_size, 'crf': crf}
        return encoded[crf]['size_bytes']

    encode(28)
    shutil.copyfile(encoded[28]['file'],destinations['archive'])
    chosen, attempts = choose_crf(encode, max_bytes=SEND_BYTES, max_crf=51)
    result = {'run': str(run_dir), 'outputs': {'archive': dict(encoded[28],file=str(destinations['archive']))},
              'send_attempts': attempts, 'presented_frames': len(frames), 'video_frames': len(times),
              'fps': fps, 'duration_s': len(times)/fps, 'recorded_duration_s': meta['t_end']-begin,
              'final_hold_s': end-meta['t_end'], 'start': frames[0]['params'], 'end': frames[-1]['params'],
              'start_hp': meta['start_hp'], 'end_hp': meta['end_hp'], 'contamination':
              json.loads((run_dir/'contamination.json').read_text(encoding='utf-8'))['verdict'],
              'blind_spots': 'Encoding captured frames does not establish deep accuracy or shallow regression.'}
    if chosen is not None:
        shutil.copyfile(encoded[chosen]['file'],destinations['send'])
        result['outputs']['send'] = dict(encoded[chosen],file=str(destinations['send']))
    else:
        result['error'] = 'No CRF 28..51 fits 48,000,000 B'
    for record in result['outputs'].values():
        record['ffprobe'] = json.loads(subprocess.run([FFPROBE,'-v','error','-count_frames','-show_entries',
            'stream=codec_type,codec_name,width,height,nb_read_frames:format=duration,size','-of','json',record['file']],
            capture_output=True,text=True,check=True).stdout)
        stream = next(s for s in record['ffprobe']['streams'] if s['codec_type'] == 'video')
        if int(stream['nb_read_frames']) != len(times):
            raise ValueError('encoded frame count differs from timeline')
    (out/'video.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if chosen is None:
        raise RuntimeError(result['error'])
    return result


def record(directory):
    from viewer import launch
    from cpu_guard import guarded_run
    from perturb_scripts import write_scripts
    write_scripts()
    return guarded_run(lambda d: launch.run(F/'viewer/scripts/perturb_video.json','timed',d,
                                            readback='record',timeout=900),directory)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('run','video'))
    ap.add_argument('run_dir', nargs='?', type=Path, default=F/'logs/perturb/video')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    directory = record(args.run_dir) if args.action == 'run' else args.run_dir
    video(directory,args.out)
