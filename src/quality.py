import numpy as np

from .common import read_json, write_csv, write_json
from .text_features import BertEncoder


def stored_float(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def check_quality(ctx):
    log = ctx.logger('quality')
    qc = ctx.cfg['quality']
    rows, negation = [], []
    encoder = None
    encoder_error = None
    try:
        encoder = BertEncoder(ctx)
    except Exception as e:
        encoder_error = repr(e)
    for row in ctx.manifest():
        sid = row['sample_id']
        checks, metrics, errors = {}, {}, []
        try:
            aligned = np.load(ctx.sample_path('aligned', sid, 'npz'))
            audit = read_json(ctx.sample_path('aligned', sid))
            units = read_json(ctx.sample_path('text_units', sid))
            receptive = np.load(ctx.sample_path('receptive', sid, 'npz'))
            rmeta = read_json(ctx.sample_path('receptive', sid))
            for m in ['text', 'audio', 'vision']:
                x = aligned[m]
                checks[m + '_finite_float16'] = bool(np.isfinite(stored_float(x)).all())
                valid = aligned[m + '_valid_mask'].astype(bool)
                metrics[m + '_max_abs'] = float(np.max(np.abs(x)))
                metrics[m + '_valid_ratio'] = float(valid.mean())
                metrics[m + '_valid_all_zero_rows'] = int((np.all(x == 0, axis=1) & valid).sum())
                if m != 'text':
                    checks[m + '_invalid_zero'] = bool(np.all(x[~valid] == 0))
                    frames = np.load(ctx.sample_path(m, sid, 'npz'))
                    checks[m + '_time_monotonic'] = bool(np.all(np.diff(frames['times']) >= 0))
                    recomputed = np.zeros_like(x)
                    # Recompute from float16 stored frame features, not the original float32 values.
                    source = stored_float(frames['features'])
                    for k, (start, end) in enumerate(aligned[m + '_pool_idx']):
                        if start < 0:
                            continue
                        segment = source[start:end + 1][frames['valid'][start:end + 1].astype(bool)]
                        if not len(segment):
                            raise ValueError('Nonempty pool index contains no valid frames')
                        mean = segment.mean(axis=0, dtype=np.float32)
                        recomputed[k] = np.r_[mean, segment.std(axis=0, dtype=np.float32)] if ctx.cfg['pooling']['method'] == 'mean_std' else mean
                    checks[m + '_pool_recompute'] = bool(np.allclose(recomputed, stored_float(x), rtol=qc['pool_rtol'], atol=qc['pool_atol']))
                    metrics[m + '_pool_max_abs_error'] = float(np.max(np.abs(recomputed - stored_float(x))))
                    if m == 'vision':
                        metrics['face_fail_ratio'] = float(1 - frames['valid'].mean())
            if encoder is None:
                raise RuntimeError('BERT recomputation unavailable: ' + str(encoder_error))
            text = encoder.encode(units)
            checks['text_recompute'] = bool(np.allclose(text, stored_float(aligned['text']), rtol=qc['text_rtol'], atol=qc['text_atol']))
            checks['batch_invariance'] = read_json(ctx.sample_path('text', sid))['batch_check_pass']
            r = stored_float(receptive['R_occ'])
            checks['R_finite_nonnegative'] = bool(np.isfinite(r).all() and np.all(r >= 0))
            checks['R_row_sum'] = bool(np.all(np.abs(r.sum(axis=1) - 1) <= ctx.cfg['receptive_field']['row_sum_tolerance']))
            checks['R_repeat'] = rmeta['repeat_max_abs_error'] is not None and rmeta['repeat_max_abs_error'] <= ctx.cfg['receptive_field']['repeat_atol']
            words = audit['words']
            speech = [w for w in words if w['has_speech']]
            timed = [w for w in speech if w['time_valid']]
            checks['word_intervals_valid'] = all(0 <= w['start'] < w['end'] for w in timed)
            checks['word_times_within_clip'] = all(w['end'] <= row['duration_s'] + ctx.cfg['media']['pts_tolerance_s'] for w in timed)
            checks['word_start_monotonic'] = all(a['start'] <= b['start'] for a, b in zip(timed, timed[1:]))
            metrics['word_align_success_rate'] = len(timed) / max(1, len(speech))
            metrics['low_conf_words'] = len(audit['quality_flags']['low_confidence_words'])
            for j, w in enumerate(words[:-1]):
                norm = w['norm'].strip('.,!?;:')
                if norm in ctx.cfg['visualize']['preferred_terms'] or norm.endswith("n't"):
                    targets = np.flatnonzero(aligned['token_word_idx'] == j + 1)
                    negation.append({'sample_id': sid, 'source_word': w['text'], 'next_word': words[j + 1]['text'],
                                     'mean_sensitivity': float(r[targets, j].mean()) if len(targets) else None})
        except Exception as e:
            errors.append(repr(e))
        failed = [name for name, passed in checks.items() if not passed]
        item = {'sample_id': sid, 'passed': bool(checks) and not failed and not errors,
                'checks': checks, 'metrics': metrics, 'failed_checks': failed, 'errors': errors}
        rows.append(item)
        log.info('%s passed=%s failed=%s errors=%s', sid, item['passed'], failed, errors)
    passed = sum(r['passed'] for r in rows)
    report = {'n_samples': len(rows), 'n_passed': passed, 'pass_rate': passed / max(1, len(rows)),
              'all_passed': passed == len(rows) and bool(rows), 'samples': rows,
              'human_alignment_review': 'required manual review; no human verification is claimed',
              'padding_validation': 'performed again on serialized exports',
              'float16_comparison': {'rtol': qc['pool_rtol'], 'atol': qc['pool_atol']}}
    write_json(ctx.path('meta', 'quality_report.json'), report)
    write_csv(ctx.path('meta', 'negation_context_check.csv'), negation,
              ['sample_id', 'source_word', 'next_word', 'mean_sensitivity'])
    from .export import summary_rows
    write_csv(ctx.path('meta', 'summary_100.csv'), summary_rows(ctx, report))
    if not report['all_passed']:
        raise ValueError(f'Quality gate: {passed}/{len(rows)} passed; see quality_report.json')
