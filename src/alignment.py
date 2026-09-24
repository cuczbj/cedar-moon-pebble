"""Deterministic interval and pooling logic; no semantic claims about weights."""
import numpy as np


def token_anchors(words, word_ids, rule='shared'):
    out = np.full((len(word_ids), 2), -1, dtype=np.float32)
    for j, word in enumerate(words):
        positions = np.flatnonzero(np.asarray(word_ids) == j)
        if not word['has_speech'] or not word['time_valid'] or not len(positions):
            continue
        start, end = word['start'], word['end']
        if rule == 'shared':
            out[positions] = [start, end]
        elif rule == 'equal':
            bounds = np.linspace(start, end, len(positions) + 1, dtype=np.float32)
            out[positions] = np.stack([bounds[:-1], bounds[1:]], axis=1)
        else:
            raise ValueError(f'Unknown subword time rule: {rule}')
    return out


def pool_frames(features, intervals, valid, anchors, word_ids, method='mean',
                extend=True, min_window=0.01):
    features = np.asarray(features, dtype=np.float32)
    intervals = np.asarray(intervals, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if len(features) != len(intervals) or len(valid) != len(features):
        raise ValueError('Frame arrays must have equal length')
    if method not in ('mean', 'mean_std'):
        raise ValueError('Only mean and mean_std pooling are supported')
    dim = features.shape[1] * (2 if method == 'mean_std' else 1)
    pooled = np.zeros((len(anchors), dim), dtype=np.float32)
    idx = np.full((len(anchors), 2), -1, dtype=np.int32)
    masks = np.zeros(len(anchors), dtype=np.int32)
    reasons = np.full(len(anchors), 3, dtype=np.int32)
    events = []
    weights = []
    for k, (start, end) in enumerate(anchors):
        weights.append({'indices': [], 'weights': []})
        if word_ids[k] < 0:
            reasons[k] = 1
            continue
        if start < 0 or end <= start:
            continue
        hit = (intervals[:, 0] < end) & (intervals[:, 1] > start)
        # Expand ONLY when there are no temporal frames, never to replace a failed face.
        if not hit.any() and extend:
            center = (start + end) / 2
            width = max(float(end - start), min_window)
            left, right = center - width / 2, center + width / 2
            hit = (intervals[:, 0] < right) & (intervals[:, 1] > left)
            events.append({'position': k, 'original': [float(start), float(end)],
                           'expanded': [float(left), float(right)], 'found': bool(hit.any())})
        candidates = np.flatnonzero(hit)
        chosen = np.flatnonzero(hit & valid)
        if not len(chosen):
            reasons[k] = 4
            continue
        if not np.all(hit[candidates[0]:candidates[-1] + 1]):
            raise ValueError('Pooling interval is noncontiguous in frame ordering')
        idx[k] = [int(candidates[0]), int(candidates[-1])]
        values = features[chosen]
        mean = values.mean(axis=0, dtype=np.float32)
        pooled[k] = np.concatenate([mean, values.std(axis=0, dtype=np.float32)]) if method == 'mean_std' else mean
        masks[k], reasons[k] = 1, 0
        weights[k] = {'indices': chosen.tolist(), 'weights': [1.0 / len(chosen)] * len(chosen)}
    return pooled, masks, reasons, idx, events, weights


def normalize_rows(raw, epsilon):
    raw = np.maximum(np.asarray(raw, np.float32), 0)
    total = raw.sum(axis=-1, keepdims=True, dtype=np.float32)
    return raw / (total + epsilon), (total[:, 0] <= epsilon)


def word_matrix(matrix, word_ids, n_words):
    result = np.zeros((n_words, n_words), dtype=np.float32)
    for j in range(n_words):
        rows = np.asarray(word_ids) == j
        if rows.any():
            result[j] = matrix[rows].mean(axis=0, dtype=np.float32)
    return result
