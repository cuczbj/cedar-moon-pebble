import gzip
import pickle
from pathlib import Path

import numpy as np

from .common import read_json, write_csv, write_json


def load_features(path):
    """Accept plain or gzip-compressed PKL; suffix intentionally follows requested schema."""
    with open(path, 'rb') as f:
        compressed = f.read(2) == b'\x1f\x8b'
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as f:
        return pickle.load(f)


def dump_features(path, data, compress):
    tmp = Path(str(path) + '.tmp')
    if compress:
        with tmp.open('wb') as file:
            with gzip.GzipFile(fileobj=file, mode='wb', mtime=0) as f:
                pickle.dump(data, f, protocol=5)
    else:
        with tmp.open('wb') as f:
            pickle.dump(data, f, protocol=5)
    tmp.replace(path)


def pad(a, shape, fill=0):
    a = np.asarray(a)
    out = np.full(shape, fill, dtype=np.float16 if a.dtype.kind == 'f' else np.int32)
    if any(x > y for x, y in zip(a.shape, shape)):
        raise ValueError('Unexpected truncation in full export')
    out[tuple(slice(0, n) for n in a.shape)] = a
    return out


def summary_rows(ctx, report):
    quality = {r['sample_id']: r for r in report['samples']}
    for row in ctx.manifest():
        sid = row['sample_id']
        q = quality.get(sid, {})
        result = {'sample_id': sid, 'video_path': row['video_path'], 'duration_s': row.get('duration_s'),
                  'fps': None, 'audio_sr': None, 'align_granularity': 'wordpiece',
                  'n_words': None, 'n_wordpieces': None, 'valid_length': None,
                  'truncated_in_compat50': None, 'dim_text': None, 'dim_audio': None, 'dim_vision': None,
                  'word_align_success_rate': None, 'audio_valid_ratio': None, 'vision_valid_ratio': None,
                  'face_fail_ratio': None, 'low_conf_words': None,
                  'status': 'ok' if q.get('passed') else 'failed',
                  'failure_reason': q.get('errors', []) + q.get('failed_checks', [])}
        for stream in row.get('streams', []):
            if stream['type'] == 'video': result['fps'] = stream.get('fps')
            if stream['type'] == 'audio': result['audio_sr'] = stream.get('sample_rate')
        result.update({k: v for k, v in q.get('metrics', {}).items() if k in result})
        p = ctx.sample_path('text_units', sid)
        if p.exists():
            u = read_json(p); L = len(u['input_ids'])
            result.update(n_words=len(u['words']), n_wordpieces=L, valid_length=L,
                          truncated_in_compat50=L > ctx.cfg['length']['compat_length'])
        p = ctx.sample_path('aligned', sid, 'npz')
        if p.exists():
            with np.load(p) as a:
                for m in ['text', 'audio', 'vision']: result['dim_' + m] = a[m].shape[1]
        yield result


def verify_export(data, tolerance):
    failures = []
    L = data['text'].shape[1]
    for i, sid in enumerate(data['sample_ids']):
        n, W = min(int(data['valid_length'][i]), L), int(data['n_words'][i])
        for key in ['text', 'audio', 'vision', 'R_occ', 'R_word']:
            if not np.isfinite(data[key][i]).all(): failures.append([sid, key, 'nonfinite'])
        for m in ['text', 'audio', 'vision']:
            mask = data[m + '_valid_mask'][i].astype(bool)
            if np.any(data[m][i][~mask] != 0): failures.append([sid, m, 'invalid not zero'])
            if mask[n:].any(): failures.append([sid, m, 'padding valid'])
        R = data['R_occ'][i].astype(np.float32)
        if np.any(R < 0) or not np.allclose(R[:n, :W].sum(axis=1), 1, atol=tolerance, rtol=0):
            failures.append([sid, 'R_occ', 'normalization'])
        if np.any(R[n:] != 0) or np.any(R[:, W:] != 0): failures.append([sid, 'R_occ', 'padding'])
    return failures


def verify_saved_frame_pools(data, frame_archive, rtol, atol):
    failures, checked = [], 0
    method = data['encoder_info']['audio']['pooling']
    for i, sid in enumerate(data['sample_ids']):
        for m in ['audio', 'vision']:
            prefix = data['frame_keys'][i] + '_' + m + '_'
            features = frame_archive[prefix + 'features'].astype(np.float32)
            valid = frame_archive[prefix + 'valid'].astype(bool)
            expected = data[m][i].astype(np.float32)
            actual = np.zeros_like(expected)
            for k, (start, end) in enumerate(data[m + '_pool_idx'][i]):
                if start < 0:
                    continue
                if end >= len(features):
                    failures.append([sid, m, 'saved frames do not cover full window'])
                    continue
                segment = features[start:end+1][valid[start:end+1]]
                if not len(segment):
                    failures.append([sid, m, 'pool has no saved valid frames'])
                    continue
                mean = segment.mean(axis=0, dtype=np.float32)
                actual[k] = np.r_[mean, segment.std(axis=0, dtype=np.float32)] if method == 'mean_std' else mean
            if not np.allclose(actual, expected, rtol=rtol, atol=atol):
                failures.append([sid, m, 'saved pool recomputation mismatch'])
        checked += 1
    return {'samples_checked': checked, 'failures': failures, 'passed': not failures}


def export(ctx):
    log = ctx.logger('s15_export')
    report = read_json(ctx.path('meta', 'quality_report.json'))
    write_csv(ctx.path('meta', 'summary_100.csv'), summary_rows(ctx, report))
    if not report['all_passed']:
        raise ValueError('Formal export refused: quality gate failed; all sample statuses retained in summary_100.csv')
    policy = read_json(ctx.path('meta', 'length_policy.json'))
    manifest = ctx.manifest()
    quality_by_id = {r['sample_id']: r for r in report['samples']}
    L, W = policy['L_max_pad'], policy['W_max']
    data = {'sample_ids': [], 'token_str': [], 'words': [], 'quality_flags': []}
    arrays, frames = {}, {}
    infos = {}
    for si, row in enumerate(manifest):
        sid = row['sample_id']
        a = np.load(ctx.sample_path('aligned', sid, 'npz'))
        audit = read_json(ctx.sample_path('aligned', sid))
        r = np.load(ctx.sample_path('receptive', sid, 'npz'))
        n, nw = len(a['text']), len(audit['words'])
        data['sample_ids'].append(sid)
        data['token_str'].append(audit['token_str'] + ['[PAD]'] * (L - n))
        data['words'].append(audit['words'])
        flags = audit['quality_flags']
        flags.update(quality_by_id[sid].get('metrics', {}))
        compat = ctx.cfg['length']['compat_length']
        flags['truncated_in_compat50'] = n > compat
        dropped = sorted({int(w) for w in a['token_word_idx'][compat:] if w >= 0})
        flags['compat_dropped_words'] = [audit['words'][j] for j in dropped]
        data['quality_flags'].append(flags)
        for key in a.files:
            value = a[key]
            fill = -1 if key.endswith('_idx') or key == 'token_time' else (2 if key.startswith('invalid_reason') else 0)
            arrays.setdefault(key, []).append(pad(value, (L,) + value.shape[1:], fill))
        for key in ['R_occ', 'R_raw', 'R_word', 'R_rollout']:
            if key in r:
                arrays.setdefault(key, []).append(pad(r[key], (W, W) if key == 'R_word' else (L, W)))
        arrays.setdefault('valid_length', []).append(n)
        arrays.setdefault('n_words', []).append(nw)
        for m in ['audio', 'vision']:
            f = np.load(ctx.sample_path(m, sid, 'npz'))
            for key in f.files:
                # float32 time coordinates preserve frame timing better than half precision.
                dtype = np.float16 if key == 'features' else (np.int32 if key == 'valid' else np.float32)
                frames[f's{si:03d}_{m}_{key}'] = f[key].astype(dtype)
            infos[m] = read_json(ctx.sample_path(m, sid))
        records = []
        for k, tok in enumerate(audit['token_str']):
            wi = int(a['token_word_idx'][k])
            top = np.argsort(-r['R_occ'][k], kind='stable')[:min(3, nw)]
            records.append({'pos': k, 'token': tok, 'word_idx': wi,
                'word': audit['words'][wi]['text'] if wi >= 0 else '',
                'start': float(a['token_time'][k, 0]), 'end': float(a['token_time'][k, 1]),
                'audio_frames': a['audio_pool_idx'][k].tolist(), 'vision_frames': a['vision_pool_idx'][k].tolist(),
                'audio_valid': int(a['audio_valid_mask'][k]), 'vision_valid': int(a['vision_valid_mask'][k]),
                'invalid_reason_audio': int(a['invalid_reason_audio'][k]), 'invalid_reason_vision': int(a['invalid_reason_vision'][k]),
                'audio_pool': audit['pool_weights']['audio'][k], 'vision_pool': audit['pool_weights']['vision'][k],
                'top3_context_words': [{'word': audit['words'][j]['text'], 'sensitivity': float(r['R_occ'][k, j])} for j in top]})
        write_csv(ctx.path('meta', 'alignment_records', sid + '.csv'), records)
    for key, values in arrays.items():
        data[key] = np.asarray(values, dtype=np.int32) if key in ('valid_length', 'n_words') else np.stack(values)
    data['encoder_info'] = {'text': {**ctx.cfg['text'], 'context': 'full sentence bidirectional',
                           'receptive_field': 'R_occ = normalized perturbation sensitivity, not semantic percentage'},
                            'audio': {**infos['audio'], 'pooling': ctx.cfg['pooling']['method']},
                            'vision': {**infos['vision'], 'pooling': ctx.cfg['pooling']['method']},
                            'storage': {'float': 'float16', 'frame_times': 'float32', 'pickle_compression': 'gzip' if ctx.cfg['export']['gzip_pickle'] else 'none'},
                            'compatibility': 'same organization only; NOT numerically equivalent to attachment 2 features'}
    data['frame_keys'] = [f's{i:03d}' for i in range(len(manifest))]
    full = ctx.path('features', 'q1_features_full.pkl')
    compat_path = ctx.path('features', 'q1_features_compat50.pkl')
    frame_path = ctx.path('features', 'q1_frames.npz')
    if not ctx.cfg['export']['include_raw']: data.pop('R_raw', None)
    if not ctx.cfg['export']['include_rollout']: data.pop('R_rollout', None)
    removed = []
    def save_all():
        dump_features(full, data, ctx.cfg['export']['gzip_pickle'])
        if policy['needs_compat']:
            compat_data = dict(data)
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.ndim >= 2 and key != 'R_word':
                    compat_data[key] = value[:, :compat]
            compat_data['valid_length_full'] = data['valid_length'].copy()
            compat_data['valid_length'] = np.minimum(data['valid_length'], compat)
            compat_data['token_str'] = [t[:compat] for t in data['token_str']]
            dump_features(compat_path, compat_data, ctx.cfg['export']['gzip_pickle'])
        else:
            log.info('No sample exceeds compat length; full file is the sole version (padding follows L_max_pad).')
        np.savez_compressed(frame_path, **frames)
        paths = [full, frame_path] + ([compat_path] if policy['needs_compat'] else [])
        return {p.name: p.stat().st_size for p in paths}
    sizes = save_all()
    budget = ctx.cfg['export']['target_mb'] * 1024 ** 2
    for optional in ['R_raw', 'R_rollout']:
        if sum(sizes.values()) <= budget: break
        if optional in data:
            data.pop(optional); removed.append(optional); sizes = save_all()
    if sum(sizes.values()) > budget and policy['needs_compat']:
        for i in range(len(manifest)):
            for m in ['audio', 'vision']:
                end = int(data[m + '_pool_idx'][i, :compat, 1].max()) + 1
                for key in ('features', 'times', 'intervals', 'valid'):
                    name = f's{i:03d}_{m}_{key}'
                    frames[name] = frames[name][:end]
        removed.append('frame_suffix_not_needed_by_compat50')
        data['encoder_info']['storage']['frame_scope'] = 'compat50 only; full frame recomputation requires retained intermediate files'
        sizes = save_all()
    full_saved = load_features(full)
    failures = verify_export(full_saved, ctx.cfg['receptive_field']['row_sum_tolerance'])
    saved_checks = {}
    with np.load(frame_path) as archive:
        versions = [('full', full_saved)] if 'frame_suffix_not_needed_by_compat50' not in removed else []
        if policy['needs_compat']:
            versions.append(('compat50', load_features(compat_path)))
        for version, saved in versions:
            saved_checks[version] = verify_saved_frame_pools(saved, archive, ctx.cfg['quality']['pool_rtol'], ctx.cfg['quality']['pool_atol'])
            failures += saved_checks[version]['failures']
    if policy['needs_compat']:
        failures += verify_export(load_features(compat_path), ctx.cfg['receptive_field']['row_sum_tolerance'])
    storage = {'files_bytes': sizes, 'total_bytes': sum(sizes.values()), 'target_bytes': budget,
               'within_budget': sum(sizes.values()) <= budget, 'removed': removed, 'serialized_checks': failures,
               'saved_frame_pool_checks': saved_checks,
               'frame_key_mapping': dict(zip(data['sample_ids'], data['frame_keys']))}
    write_json(ctx.path('meta', 'export_report.json'), storage)
    report['serialized_export'] = storage
    write_json(ctx.path('meta', 'quality_report.json'), report)
    for name, size in sizes.items(): log.info('%s %.3f MiB', name, size / 1024 ** 2)
    log.info('TOTAL %.3f MiB; removed=%s', sum(sizes.values()) / 1024 ** 2, removed)
    if failures or not storage['within_budget']:
        raise ValueError('Export verification/size gate failed; see export_report.json; outputs are not accepted deliverables')
