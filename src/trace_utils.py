"""Sensitivity-based redistribution, NOT a causal or semantic decomposition.

Mean-pooling projection preserves supplied nonnegative mass where mappings exist.
For mean+std features an exact attribution requires downstream derivatives;
the wrapper refuses to silently use mean weights for the standard deviation.
"""
import numpy as np


def _importance(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Importance must be a finite nonnegative vector; handle signed contributions separately')
    return a


def text_position_to_words(a_text, R_occ, n_words):
    a = _importance(a_text)
    r = np.asarray(R_occ, dtype=np.float64)
    if r.ndim != 2 or len(a) != r.shape[0] or not 0 <= n_words <= r.shape[1]:
        raise ValueError('Incompatible position/word dimensions')
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError('Invalid receptive field matrix')
    return a @ r[:, :n_words]


def words_to_time_segments(s, words, drop_no_speech=True, normalize=True):
    s = _importance(s)
    if len(s) != len(words):
        raise ValueError('Word count mismatch')
    segments, excluded, unmapped = [], [], 0.0
    for value, word in zip(s, words):
        eligible = (word.get('has_speech', True) and word.get('time_valid', False)
                    and word.get('start', -1) >= 0 and word.get('end', -1) > word.get('start', -1))
        if eligible:
            segments.append((word['start'], word['end'], float(value), word['text']))
        else:
            unmapped += value
            if not drop_no_speech:
                excluded.append({'word': word['text'], 'score': float(value), 'time': None})
    mapped = sum(x[2] for x in segments)
    if normalize and mapped > 0:
        segments = [(a, b, score / mapped, w) for a, b, score, w in segments]
    return {'segments': segments, 'unmapped_mass': float(unmapped),
            'mapped_mass_before_normalization': mapped, 'unmapped_words': excluded,
            'normalized': normalize}


def modality_position_to_frames(a_mod, pool_idx, frame_valid, frame_times):
    a = _importance(a_mod)
    idx = np.asarray(pool_idx)
    valid = np.asarray(frame_valid, dtype=bool)
    times = np.asarray(frame_times)
    if idx.shape != (len(a), 2) or len(times) != len(valid):
        raise ValueError('Pooling/frame dimension mismatch')
    importance = np.zeros(len(valid), dtype=np.float64)
    unmapped = 0.0
    for score, (start, end) in zip(a, idx):
        if start < 0:
            unmapped += score
            continue
        if end < start or end >= len(valid):
            raise ValueError('Frame index out of bounds')
        frames = np.arange(start, end + 1, dtype=int)
        frames = frames[valid[frames]]
        if len(frames):
            importance[frames] += score / len(frames)
        else:
            unmapped += score
    return {'frame_indices': np.arange(len(valid), dtype=np.int32), 'times': times,
            'importance': importance, 'unmapped_mass': float(unmapped)}


def trace_sample(sample, a_text, a_audio, a_vision, top_k=5, normalize_words=True):
    if sample.get('pooling', sample.get('encoder_info', {}).get('audio', {}).get('pooling', 'mean')) != 'mean':
        raise ValueError('Exact frame redistribution is implemented only for mean pooling, not std features')
    n = int(sample['n_words'])
    word_scores = text_position_to_words(a_text, sample['R_occ'], n)
    text = words_to_time_segments(word_scores, sample['words'][:n], normalize=normalize_words)
    text['position_mass_not_assigned_to_words'] = float(np.sum(a_text) - np.sum(word_scores))
    result = {'text': text, 'interpretation': 'normalized perturbation sensitivity projection, not causal attribution'}
    for m, a in [('audio', a_audio), ('vision', a_vision)]:
        r = modality_position_to_frames(a, sample[m + '_pool_idx'],
                                        sample[m + '_frame_valid'], sample[m + '_frame_times'])
        positive = np.flatnonzero(r['importance'] > 0)
        top = positive[np.argsort(-r['importance'][positive], kind='stable')[:top_k]]
        r['top_frames'] = [{'index': int(i), 'time': float(r['times'][i]),
                            'score': float(r['importance'][i])} for i in top]
        result[m] = r
    return result
