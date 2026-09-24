from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np

from .alignment import pool_frames, token_anchors
from .common import read_json, record_model, run_command, sha256, write_csv, write_json
from .media import audio_features, preprocess, probe, visual_features
from .text_features import BertEncoder, build_words, match_alignment


def s01_manifest(ctx):
    import openpyxl
    paths = list(ctx.root.glob(ctx.cfg['label_glob']))
    if len(paths) != 1:
        raise ValueError(f'Expected one label table, found {paths}')
    wb = openpyxl.load_workbook(paths[0], read_only=True, data_only=True)
    iterator = iter(wb.active.values)
    header = [str(x).strip() for x in next(iterator)]
    required = ['video_id', 'clip_id', 'text', 'label', 'annotation']
    if not set(required) <= set(header):
        raise ValueError(f'Label columns: {header}')
    rows = []
    seen = set()
    for values in iterator:
        r = dict(zip(header, values))
        if all(x is None for x in values):
            continue
        vid = str(r['video_id'])
        clip = str(int(r['clip_id'])) if isinstance(r['clip_id'], (int, float)) else str(r['clip_id'])
        sid = vid + '$_$' + clip
        if sid in seen:
            raise ValueError('Duplicate official sample: ' + sid)
        seen.add(sid)
        video = paths[0].parent / vid / (clip + '.mp4')
        rows.append({'sample_id': sid, 'video_id': vid, 'clip_id': clip, 'video_path': str(video.resolve()),
                     'text': str(r['text']), 'label': float(r['label']), 'annotation': str(r['annotation']),
                     'source_exists': video.is_file(), 'status': 'pending' if video.is_file() else 'missing_video'})
    wb.close()
    if len(rows) != ctx.cfg['expected_samples']:
        raise ValueError(f'Expected {ctx.cfg["expected_samples"]} official rows, found {len(rows)}')
    videos = list(paths[0].parent.rglob('*.mp4'))
    known = {Path(r['video_path']).resolve() for r in rows}
    write_json(ctx.path('meta', 'manifest_audit.json'), {'official_count': len(rows), 'source_video_groups': len({r['video_id'] for r in rows}),
               'video_count': len(videos), 'unmatched_videos': [str(x) for x in videos if x.resolve() not in known],
               'missing_videos': [r['sample_id'] for r in rows if not r['source_exists']], 'label_sha256': sha256(paths[0])})
    if ctx.limit:
        rows = rows[:ctx.limit]
    write_json(ctx.path('meta', 'manifest.json'), rows)
    write_csv(ctx.path('meta', 'manifest.csv'), rows)
    ctx.logger('s01_manifest').info('Retained %d samples; groups=%d', len(rows), len({r['video_id'] for r in rows}))


def s02_probe_media(ctx):
    def work(row):
        info = probe(row['video_path'], ctx.cfg['media'])
        write_json(ctx.sample_path('probe', row['sample_id']), info)
    ctx.each('s02_probe_media', work)
    rows = ctx.manifest()
    for row in rows:
        p = ctx.sample_path('probe', row['sample_id'])
        if p.exists():
            meta = read_json(p)
            row.update(duration_s=meta['duration_s'], streams=meta['streams'], vfr=meta['vfr'])
    write_json(ctx.path('meta', 'manifest.json'), rows)
    write_csv(ctx.path('meta', 'manifest.csv'), rows)


def s03_preprocess_av(ctx):
    ctx.each('s03_preprocess_av', lambda row: preprocess(ctx, row))


def s04_text_units(ctx):
    encoder = BertEncoder(ctx, load_weights=False)
    issues = []
    def work(row):
        words, align_units, problems = build_words(row['text'], ctx.cfg['text']['normalize_unicode'], ctx.cfg['text']['alignment_overrides'])
        tok = encoder.tokenize(words)
        for j, w in enumerate(words):
            if j not in tok['token_word_idx']:
                problems.append({'word': w['text'], 'reason': 'no WordPiece assigned', 'rule': 'fatal mapping error'})
        issues.extend({'sample_id': row['sample_id'], **x} for x in problems)
        if any(j not in tok['token_word_idx'] for j in range(len(words))):
            raise ValueError('Tokenizer lost official word')
        write_json(ctx.sample_path('text_units', row['sample_id']), {**tok, 'raw_text': row['text'],
                   'words': words, 'alignment_units': align_units, 'alignment_text': ' '.join(x['text'] for x in align_units)})
    ctx.each('s04_text_units', work)
    write_csv(ctx.path('meta', 'text_mapping_issues.csv'), issues, ['sample_id', 'word', 'reason', 'rule'])


def s05_force_align(ctx):
    cfg = ctx.cfg['alignment']
    low_rows = []
    if cfg['backend'] == 'whisperx':
        import torch
        # Only import the alignment module: never load or run an ASR transcription model.
        from whisperx.alignment import load_align_model, align
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if ctx.cfg['device'] == 'auto' else ctx.cfg['device']
        model, metadata = load_align_model(language_code=cfg['language'], device=device, model_name=cfg['model'])
        import hashlib
        h = hashlib.sha256()
        for name, weight in model.state_dict().items():
            h.update(name.encode()); h.update(weight.detach().cpu().contiguous().numpy().tobytes())
        record_model(ctx, 'forced_alignment', {'backend': 'whisperx', 'name': cfg['model'], 'state_dict_sha256': h.hexdigest()})
    elif cfg['backend'] != 'mfa':
        raise ValueError('alignment.backend must be whisperx or mfa')
    def work(row):
        import soundfile as sf
        sid = row['sample_id']
        units = read_json(ctx.sample_path('text_units', sid))
        pre = read_json(ctx.sample_path('preprocess', sid))
        waveform, sr = sf.read(pre['audio_path'], dtype='float32')
        if sr != ctx.cfg['media']['sample_rate']:
            raise ValueError('Unexpected waveform sample rate')
        if cfg['backend'] == 'whisperx':
            if sr != 16000:
                raise ValueError('WhisperX alignment requires 16-kHz audio')
            result = align([{'start': 0.0, 'end': len(waveform) / sr, 'text': units['alignment_text']}],
                           model, metadata, waveform, device, return_char_alignments=False,
                           interpolate_method='ignore')
            returned = result['word_segments']
        else:
            from praatio import textgrid
            txt = ctx.sample_path('mfa', sid, 'lab')
            txt.write_text(units['alignment_text'], encoding='utf-8')
            grid = ctx.sample_path('mfa', sid, 'TextGrid')
            run_command([cfg['mfa_executable'], 'align_one', pre['audio_path'], txt,
                         cfg['mfa_dictionary'], cfg['mfa_acoustic_model'], grid])
            tg = textgrid.openTextgrid(str(grid), includeEmptyIntervals=False)
            tier = next(tg.getTier(n) for n in tg.tierNames if 'word' in n.lower())
            returned = [{'word': e.label, 'start': e.start, 'end': e.end,
                         'score': cfg['mfa_unknown_score']} for e in tier.entries if e.label.strip()]
        words, low, audit = match_alignment(units['words'], units['alignment_units'], returned, cfg['min_score'])
        low_rows.extend({'sample_id': sid, **x} for x in low)
        write_json(ctx.sample_path('alignment', sid), {'words': words, 'unit_mapping': audit,
                   'raw_aligner_words': returned, 'low_confidence': low, 'backend': cfg['backend']})
    ctx.each('s05_force_align', work)
    write_csv(ctx.path('meta', 'low_confidence_alignment.csv'), low_rows,
              ['sample_id', 'word_idx', 'word', 'score', 'reason'])


def s06_text_encode(ctx):
    encoder = BertEncoder(ctx)
    def work(row):
        sid = row['sample_id']
        unit = read_json(ctx.sample_path('text_units', sid))
        values = encoder.encode(unit)
        delta = encoder.batch_check(unit, ctx.cfg['text']['batch_check_length_difference'])
        write_json(ctx.sample_path('text', sid), {'batch_max_abs_error': delta,
                   'batch_check_pass': delta < ctx.cfg['text']['batch_tolerance'], 'shape': list(values.shape)})
        if delta >= ctx.cfg['text']['batch_tolerance']:
            raise ValueError(f'Batch padding invariance failed: {delta}')
        np.save(ctx.sample_path('text', sid, 'npy'), values)
    ctx.each('s06_text_encode', work)


def s06b_text_receptive_field(ctx):
    from scipy.stats import spearmanr
    encoder = BertEncoder(ctx)
    cfg = ctx.cfg['receptive_field']
    agreements = []
    if cfg['gradient']:
        raise ValueError('R_grad optional Jacobian method is not enabled: vector-output gradient needs an explicit scalar/Jacobian norm definition. Use implemented occlusion and rollout.')
    def work(row):
        sid = row['sample_id']
        unit = read_json(ctx.sample_path('text_units', sid))
        n = len(unit['words'])
        r = encoder.occlusion(unit, n, cfg['epsilon'])
        delta = None
        if cfg['repeat_check']:
            repeated = encoder.occlusion(unit, n, cfg['epsilon'])
            delta = float(np.max(np.abs(r['R_occ'] - repeated['R_occ'])))
        if cfg['rollout']:
            r['R_rollout'] = encoder.rollout(unit, n, cfg['epsilon'])
            for k, (a, b) in enumerate(zip(r['R_occ'], r['R_rollout'])):
                corr = float(spearmanr(a, b).statistic) if n > 1 and np.ptp(a) > 0 and np.ptp(b) > 0 else None
                agreements.append({'sample_id': sid, 'position': k, 'method': 'occ_vs_rollout', 'spearman': corr})
        np.savez_compressed(ctx.sample_path('receptive', sid, 'npz'), **r)
        write_json(ctx.sample_path('receptive', sid), {'repeat_max_abs_error': delta,
                   'degenerate_rows': r['degenerate_rows'], 'interpretation': 'normalized MASK-induced representation sensitivity; not a semantic percentage or causal attribution'})
        if len(r['degenerate_rows']) and ctx.cfg['quality']['fail_on_degenerate_receptive_rows']:
            raise ValueError('Zero-sensitivity rows have no defensible unit-sum distribution')
        if delta is not None and delta > cfg['repeat_atol']:
            raise ValueError('Repeated occlusion did not reproduce')
    ctx.each('s06b_text_receptive_field', work)
    write_csv(ctx.path('meta', 'receptive_field_agreement.csv'), agreements,
              ['sample_id', 'position', 'method', 'spearman'])
    vals = [r['spearman'] for r in agreements if r['spearman'] is not None]
    write_json(ctx.path('meta', 'receptive_field_agreement_summary.json'), {'n_rows': len(vals),
               'mean': float(np.mean(vals)) if vals else None,
               'quantiles': np.quantile(vals, [0, .25, .5, .75, 1]).tolist() if vals else [],
               'R_grad': 'not implemented (optional)', 'rollout_is_not_ground_truth': True})


def s07_audio_features(ctx):
    ctx.each('s07_audio_features', lambda row: audio_features(ctx, row))


def s08_visual_features(ctx):
    ctx.each('s08_visual_features', lambda row: visual_features(ctx, row))


def s09_11_align_pool_mask(ctx):
    def work(row):
        sid = row['sample_id']
        units = read_json(ctx.sample_path('text_units', sid))
        aligned = read_json(ctx.sample_path('alignment', sid))
        words = aligned['words']
        ids = np.asarray(units['token_word_idx'], np.int32)
        anchors = token_anchors(words, ids, ctx.cfg['pooling']['subword_time'])
        text = np.load(ctx.sample_path('text', sid, 'npy')).astype(np.float32)
        result = {'text': text, 'text_valid_mask': np.ones(len(ids), np.int32),
                  'token_word_idx': ids, 'token_time': anchors}
        audit = {'words': words, 'token_str': units['token_str'], 'quality_flags': {}, 'pool_weights': {}}
        for m in ['audio', 'vision']:
            frames = np.load(ctx.sample_path(m, sid, 'npz'))
            cfg = ctx.cfg['pooling']
            width = cfg['min_audio_window_s'] if m == 'audio' else cfg['min_visual_window_s']
            # At least one typical observed frame duration; never jump to a distant frame.
            width = max(width, float(np.median(frames['intervals'][:, 1] - frames['intervals'][:, 0])))
            values, valid, reasons, idx, events, weights = pool_frames(frames['features'], frames['intervals'], frames['valid'],
                    anchors, ids, cfg['method'], cfg['extend_empty_window'], width)
            result.update({m: values, m + '_valid_mask': valid, 'invalid_reason_' + m: reasons, m + '_pool_idx': idx})
            audit['quality_flags'][m + '_window_extensions'] = events
            audit['pool_weights'][m] = weights
        audit['quality_flags']['low_confidence_words'] = aligned['low_confidence']
        np.savez_compressed(ctx.sample_path('aligned', sid, 'npz'), **result)
        write_json(ctx.sample_path('aligned', sid), audit)
    ctx.each('s09_11_align_pool_mask', work)


def s12_length_policy(ctx):
    rows, missing = [], []
    for row in ctx.manifest():
        p = ctx.sample_path('text_units', row['sample_id'])
        if not p.exists():
            missing.append(row['sample_id']); continue
        u = read_json(p)
        L, W = len(u['input_ids']), len(u['words'])
        limit = ctx.cfg['length']['compat_length']
        dropped_ids = sorted({i for i in u['token_word_idx'][limit:] if i >= 0})
        align_path = ctx.sample_path('alignment', row['sample_id'])
        words = read_json(align_path)['words'] if align_path.exists() else u['words']
        rows.append({'sample_id': row['sample_id'], 'n_words': W, 'n_wordpieces': L,
                     'truncated_in_compat50': L > limit, 'dropped_words': [words[i] for i in dropped_ids]})
    write_csv(ctx.path('meta', 'length_stats.csv'), rows)
    if missing:
        raise ValueError(f'Tokenization missing for {missing}')
    L = max(r['n_wordpieces'] for r in rows)
    multiple = ctx.cfg['length']['pad_multiple']
    info = {'N': len(rows), 'L_max_pad': math.ceil(L / multiple) * multiple,
            'W_max': max(r['n_words'] for r in rows), 'needs_compat': L > ctx.cfg['length']['compat_length'],
            'wordpiece_quantiles': np.quantile([r['n_wordpieces'] for r in rows], [0, .25, .5, .75, .9, 1]).tolist(),
            'word_quantiles': np.quantile([r['n_words'] for r in rows], [0, .25, .5, .75, .9, 1]).tolist()}
    write_json(ctx.path('meta', 'length_policy.json'), info)
    ctx.logger('s12_length_policy').info('%s', info)


def s13_14_quality_check(ctx):
    from .quality import check_quality
    check_quality(ctx)


def s15_export(ctx):
    from .export import export
    export(ctx)


def s16_visualize(ctx):
    from .visualize import visualize
    visualize(ctx)
