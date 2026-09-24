from __future__ import annotations

import csv
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from .common import read_json, run_command, write_json


def probe(path, cfg):
    import av
    result = {'streams': [], 'video_pts': [], 'audio_intervals': [], 'warnings': []}
    with av.open(str(path)) as container:
        for stream in container.streams:
            if stream.type not in ('video', 'audio'):
                continue
            info = {'type': stream.type, 'time_base': str(stream.time_base),
                    'start_time': float(stream.start_time * stream.time_base) if stream.start_time is not None else None,
                    'duration': float(stream.duration * stream.time_base) if stream.duration is not None else None,
                    'codec': stream.codec_context.name}
            if stream.type == 'video':
                info.update(width=stream.width, height=stream.height,
                            fps=float(stream.average_rate) if stream.average_rate else None)
            else:
                info.update(sample_rate=stream.sample_rate, channels=stream.codec_context.channels)
            result['streams'].append(info)
    for kind in ['video', 'audio']:
        with av.open(str(path)) as container:
            streams = [s for s in container.streams if s.type == kind]
            if not streams:
                result['warnings'].append(f'missing_{kind}_stream')
                continue
            for frame in container.decode(streams[0]):
                if frame.pts is None:
                    raise ValueError(f'{kind} frame without PTS')
                start = float(frame.pts * frame.time_base)
                if kind == 'video':
                    result['video_pts'].append(start)
                else:
                    result['audio_intervals'].append([start, start + frame.samples / frame.sample_rate])
    pts = np.array(result['video_pts'], dtype=np.float64)
    audio = np.array(result['audio_intervals'], dtype=np.float64).reshape(-1, 2)
    diff = np.diff(pts)
    if len(diff) and np.any(diff <= 0):
        raise ValueError('Video PTS not strictly increasing')
    if len(audio) > 1 and np.any(np.diff(audio[:, 0]) <= 0):
        raise ValueError('Audio PTS not strictly increasing')
    last_step = float(np.median(diff)) if len(diff) else 0
    v_end = float(pts[-1] + last_step) if len(pts) else 0
    a_end = float(audio[-1, 1]) if len(audio) else 0
    result['duration_s'] = max(v_end, a_end)
    result['time_origin'] = 0.0
    result['time_reference'] = 'decoded presentation timestamps in source clip; audio resampled onto same zero origin'
    result['vfr'] = bool(len(diff) and np.max(np.abs(diff - last_step)) > cfg['pts_tolerance_s'])
    result['av_end_difference_s'] = v_end - a_end
    result['video_intervals'] = [[float(p), float(pts[i + 1] if i + 1 < len(pts) else v_end)] for i, p in enumerate(pts)]
    result['audio_gaps'] = [[float(a), float(b)] for a, b in zip(audio[:-1, 1], audio[1:, 0])
                            if b - a > cfg['pts_tolerance_s']]
    if result['duration_s'] < cfg['short_clip_seconds']:
        result['warnings'].append('short_clip')
    if len(diff) and np.any(diff > last_step * cfg['frame_gap_factor']):
        result['warnings'].append('video_pts_gap')
    return result


def preprocess(ctx, row):
    import av
    import soundfile as sf
    sid = row['sample_id']
    meta = read_json(ctx.sample_path('probe', sid))
    folder = ctx.path('intermediate', 'media', sid)
    folder.mkdir(parents=True, exist_ok=True)
    frame_dir = folder / 'frames'
    if shutil.disk_usage(folder).free < ctx.cfg['media']['min_free_mb'] * 1024**2:
        raise OSError('Insufficient disk reserve for preprocessing')
    count = 0
    if ctx.cfg['media']['save_frames']:
        frame_dir.mkdir(exist_ok=True)
        with av.open(row['video_path']) as c:
            for count, frame in enumerate(c.decode(video=0), 1):
                expected = meta['video_pts'][count - 1]
                if abs(float(frame.pts * frame.time_base) - expected) > ctx.cfg['media']['pts_tolerance_s']:
                    raise ValueError('Decode/probe frame PTS mismatch')
                image_path = frame_dir / f'{count:06d}.{ctx.cfg["media"]["image_format"]}'
                options = {'quality': ctx.cfg['media']['jpeg_quality']} if image_path.suffix.lower() in ('.jpg', '.jpeg') else {}
                frame.to_image().save(image_path, **options)
        if count != len(meta['video_pts']):
            raise ValueError('Decoded frame count differs from probe')
    sr = ctx.cfg['media']['sample_rate']
    chunks = []
    with av.open(row['video_path']) as c:
        if not c.streams.audio:
            raise ValueError('No audio; keep sample manifest, do not fabricate waveform')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=sr)
        for frame in c.decode(audio=0):
            for output in resampler.resample(frame):
                if output.pts is None:
                    raise ValueError('Resampler lost source timestamps')
                chunks.append((round(float(output.pts * output.time_base) * sr), output.to_ndarray().reshape(-1)))
        for output in resampler.resample(None):
            chunks.append((round(float(output.pts * output.time_base) * sr), output.to_ndarray().reshape(-1)))
    if not chunks:
        raise ValueError('No decoded audio samples')
    size = max(max(0, start + len(values)) for start, values in chunks)
    signal = np.zeros(size, dtype=np.float32)
    coverage = np.zeros(size, dtype=bool)
    for start, values in chunks:
        if start < 0:
            values, start = values[-start:], 0
        signal[start:start + len(values)] = values
        coverage[start:start + len(values)] = True
    sf.write(folder / 'audio.wav', signal, sr, subtype='FLOAT')
    np.savez_compressed(folder / 'audio_coverage.npz', valid=coverage)
    write_json(ctx.sample_path('preprocess', sid), {'sample_rate': sr, 'audio_path': str(folder / 'audio.wav'),
               'audio_time_origin': 0.0, 'n_samples': size, 'frame_dir': str(frame_dir),
               'n_video_frames': len(meta['video_pts']), 'audio_gap_samples': int((~coverage).sum()),
               'decoder': 'PyAV/FFmpeg', 'source_pts_preserved': True})


def audio_features(ctx, row):
    import opensmile
    sid = row['sample_id']
    pre = read_json(ctx.sample_path('preprocess', sid))
    ac = ctx.cfg['audio']
    feature_set = str((ctx.base / ac['config_override']).resolve()) if ac['config_override'] else getattr(opensmile.FeatureSet, ac['feature_set'])
    from opensmile.core.config import config as smile_config
    from .common import sha256
    with tempfile.TemporaryDirectory(prefix='q1_smile_') as temp:
        # The native openSMILE wrapper encodes configuration paths as ASCII.
        # Copy only small bundled configs, leaving installed packages untouched.
        config_root = Path(opensmile.__file__).parent / 'core' / smile_config.CONFIG_ROOT
        local_root = Path(temp) / 'config'
        shutil.copytree(config_root.resolve(), local_root)
        class LocalConfigSmile(opensmile.Smile):
            @property
            def default_config_root(self):
                return str(local_root)
        smile = LocalConfigSmile(feature_set=feature_set, feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                                 sampling_rate=pre['sample_rate'], resample=False)
        config_hash = sha256(smile.config_path)
        config_name = smile.config_name
        df = smile.process_file(pre['audio_path'])
    values = df.to_numpy(dtype=np.float32, copy=True)
    start = df.index.get_level_values('start').total_seconds().to_numpy(dtype=np.float32)
    end = df.index.get_level_values('end').total_seconds().to_numpy(dtype=np.float32)
    if not np.isfinite(end).all() or np.any(end <= start):
        raise ValueError('openSMILE returned invalid frame intervals')
    if len(start) > 1 and abs(float(np.median(np.diff(start))) * 1000 - ac['hop_ms']) > ac['hop_tolerance_ms']:
        raise ValueError('Actual openSMILE hop differs from configured hop; provide a matching custom config')
    coverage = np.load(Path(pre['audio_path']).with_name('audio_coverage.npz'))['valid']
    # Exclude any short window intersecting missing decoded audio; silence is still valid.
    valid = np.isfinite(values).all(axis=1)
    for i, (s, e) in enumerate(zip(start, end)):
        lo, hi = max(0, int(s * pre['sample_rate'])), min(len(coverage), int(np.ceil(e * pre['sample_rate'])))
        valid[i] &= hi > lo and bool(coverage[lo:hi].all())
    values[~valid] = 0
    np.savez_compressed(ctx.sample_path('audio', sid, 'npz'), features=values,
                        intervals=np.stack([start, end], axis=1), times=(start + end) / 2,
                        valid=valid.astype(np.int32))
    write_json(ctx.sample_path('audio', sid), {'columns': list(df.columns), 'dimension': values.shape[1],
               'feature_set': ac['feature_set'], 'hop_ms': ac['hop_ms'],
               'reported_window_ms': np.unique(np.round((end - start) * 1000, 3)).tolist(),
               'context': 'short-window descriptors; stock eGeMAPS includes smoothing and multiple internal windows; frame pooling projection is not raw-waveform attribution',
               'config_name': config_name, 'config_sha256': config_hash})


def visual_features(ctx, row):
    sid = row['sample_id']
    pre = read_json(ctx.sample_path('preprocess', sid))
    probe_info = read_json(ctx.sample_path('probe', sid))
    vc = ctx.cfg['vision']
    exe = (ctx.base / vc['executable']).resolve()
    if not exe.is_file():
        raise FileNotFoundError(f'OpenFace executable not found: {exe}; no synthetic visual features will be generated')
    dest = ctx.path('intermediate', 'openface', sid)
    dest.mkdir(parents=True, exist_ok=True)
    # One sample's frames at a time; ASCII temp path also avoids OpenCV 4's
    # Windows Unicode imread limitation. Originals are never moved or deleted.
    import av
    with tempfile.TemporaryDirectory(prefix='q1_openface_') as temp:
        scratch = Path(temp)
        input_dir, output_dir = scratch / 'frames', scratch / 'result'
        input_dir.mkdir(); output_dir.mkdir()
        count = 0
        with av.open(row['video_path']) as c:
            for count, frame in enumerate(c.decode(video=0), 1):
                if shutil.disk_usage(scratch).free < ctx.cfg['media']['min_free_mb'] * 1024**2:
                    raise OSError('Insufficient disk reserve; temporary frame cache released')
                pts = float(frame.pts * frame.time_base)
                if abs(pts - probe_info['video_pts'][count - 1]) > ctx.cfg['media']['pts_tolerance_s']:
                    raise ValueError('Visual decode PTS mismatch')
                image_path = input_dir / f'{count:06d}.{ctx.cfg["media"]["image_format"]}'
                options = {'quality': ctx.cfg['media']['jpeg_quality']} if image_path.suffix.lower() in ('.jpg', '.jpeg') else {}
                frame.to_image().save(image_path, **options)
        if count != len(probe_info['video_pts']):
            raise ValueError('Visual decode count mismatch')
        args = [exe, '-fdir', input_dir, '-out_dir', output_dir, '-of', 'features.csv', '-aus', '-pose', '-gaze']
        if vc['landmark_model']:
            landmark_model = exe.parent / vc['landmark_model']
            if not landmark_model.is_file():
                raise FileNotFoundError(f'Configured landmark model missing: {landmark_model}')
            # Relative model paths are interpreted from the OpenFace installation cwd.
            args.extend(['-mloc', vc['landmark_model']])
        if vc['au_mode'] == 'static':
            args.append('-au_static')
        elif vc['au_mode'] != 'dynamic':
            raise ValueError('au_mode must be static or dynamic')
        stdout = run_command(args, cwd=exe.parent)
        (dest / 'run.log').write_text(stdout, encoding='utf-8')
        csv_files = list(output_dir.glob('*.csv'))
        if len(csv_files) != 1:
            raise ValueError(f'Expected one OpenFace CSV, found {len(csv_files)}')
        shutil.copyfile(csv_files[0], dest / 'features.csv')
    files = list(dest.glob('*.csv'))
    if len(files) != 1:
        raise ValueError(f'Expected one OpenFace CSV, found {len(files)}')
    with files[0].open(encoding='utf-8-sig', newline='') as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    expected = len(probe_info['video_pts'])
    if len(rows) != expected:
        raise ValueError(f'OpenFace rows {len(rows)} != decoded frames {expected}; cannot assign PTS safely')
    if [int(r['frame']) for r in rows] != list(range(1, expected + 1)):
        raise ValueError('OpenFace frame indices are missing or reordered')
    columns = [k for k in rows[0] if re.match(vc['columns_regex'], k)]
    if not columns or not any(k.startswith('AU') for k in columns):
        raise ValueError('OpenFace CSV contains no configured AU features')
    features = np.asarray([[float(r[k]) for k in columns] for r in rows], dtype=np.float32)
    valid = np.array([int(r['success']) == 1 and float(r['confidence']) >= vc['min_confidence'] for r in rows])
    valid &= np.isfinite(features).all(axis=1)
    features[~valid] = 0
    np.savez_compressed(ctx.sample_path('vision', sid, 'npz'), features=features,
                        intervals=np.asarray(probe_info['video_intervals'], np.float32),
                        times=np.asarray(probe_info['video_pts'], np.float32), valid=valid.astype(np.int32))
    write_json(ctx.sample_path('vision', sid), {'columns': columns, 'dimension': len(columns),
               'au_mode': vc['au_mode'], 'fps_source': 'decoded source PTS; OpenFace CSV timestamps ignored',
               'landmark_model': vc['landmark_model'], 'analysis_image_format': ctx.cfg['media']['image_format'],
               'jpeg_quality': ctx.cfg['media']['jpeg_quality'],
               'context': 'per-image static AU estimates with sequence tracking' if vc['au_mode'] == 'static'
                          else 'per-video normalization and sequence tracking; global dependence',
               'attribution_limit': 'pooling maps only to observed frame descriptors, not an exact image-pixel attribution'})
