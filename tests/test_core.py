import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alignment import normalize_rows, pool_frames, token_anchors, word_matrix
from src.trace_utils import (modality_position_to_frames, text_position_to_words,
                            trace_sample, words_to_time_segments)
from src.text_features import build_words, match_alignment
from src.export import load_features, pad


def word(text, start=0, end=1, speech=True, valid=True):
    return {'text': text, 'norm': text.lower(), 'has_speech': speech, 'time_valid': valid,
            'start': start, 'end': end, 'score': .9}


def test_subwords_share_or_split_without_special_times():
    w = [word('playing', 1, 3), word('!', speech=False)]
    ids = [-1, 0, 0, 1, -1]
    np.testing.assert_equal(token_anchors(w, ids)[1:3], [[1, 3], [1, 3]])
    np.testing.assert_equal(token_anchors(w, ids, 'equal')[1:3], [[1, 2], [2, 3]])
    assert np.all(token_anchors(w, ids)[[0, 3, 4]] == -1)


def test_pooling_skips_failed_frames_and_reconstructs_mass():
    f = np.array([[2., 4.], [999, 999], [6., 8.]], np.float32)
    intervals = np.array([[0, 1], [1, 2], [2, 3]], np.float32)
    out, valid, reason, idx, _, weights = pool_frames(f, intervals, [1, 0, 1],
                                                    [[0, 3], [1, 2], [-1, -1]], [0, 1, -1])
    np.testing.assert_equal(out[0], [4, 6])
    np.testing.assert_equal(reason, [0, 4, 1])
    assert weights[0] == {'indices': [0, 2], 'weights': [.5, .5]}
    r = modality_position_to_frames([.6, .3, .1], idx, [1, 0, 1], [.5, 1.5, 2.5])
    np.testing.assert_allclose(r['importance'], [.3, 0, .3])
    assert r['unmapped_mass'] == pytest.approx(.4)


def test_overlapping_tokens_add_and_no_previous_frame_imputation():
    r = modality_position_to_frames([.4, .6], [[0, 1], [0, 1]], [1, 1], [0, 1])
    np.testing.assert_allclose(r['importance'], [.5, .5])
    f = np.ones((2, 1), np.float32)
    out, valid, reason, *_ = pool_frames(f, [[0, 1], [1, 2]], [1, 0], [[1, 2]], [0])
    assert valid[0] == 0 and reason[0] == 4 and out[0, 0] == 0


def test_window_expansion_records_event_but_not_distant_nearest_frame():
    out, valid, _, _, events, _ = pool_frames(np.ones((1, 1)), [[0, .1]], [1], [[.11, .12]], [0], min_window=.1)
    assert valid[0] == 1 and len(events) == 1
    _, valid, reason, *_ = pool_frames(np.ones((1, 1)), [[0, .1]], [1], [[10, 11]], [0], min_window=.1)
    assert valid[0] == 0 and reason[0] == 4


def test_mean_std_is_exact_but_trace_refuses_false_linear_attribution():
    out, *_ = pool_frames(np.array([[2.], [6.]]), [[0, 1], [1, 2]], [1, 1], [[0, 2]], [0], method='mean_std')
    np.testing.assert_equal(out, [[4, 2]])
    with pytest.raises(ValueError, match='mean pooling'):
        trace_sample({'pooling': 'mean_std'}, [], [], [])


def test_word_projection_retains_unmapped_punctuation_mass():
    r = np.array([[.25, .75], [.8, .2]])
    s = text_position_to_words([1, 0], r, 2)
    result = words_to_time_segments(s, [word('hi', 1, 2), word('!', speech=False)], drop_no_speech=False)
    assert result['unmapped_mass'] == .75
    assert result['segments'][0][2] == 1
    assert len(result['unmapped_words']) == 1


def test_unaligned_word_is_not_assigned_a_guessed_time():
    words = [word('one', valid=False), word('two', valid=False)]
    units = [{'text': 'one', 'word_idx': 0}, {'text': 'two', 'word_idx': 1}]
    w, low, audit = match_alignment(words, units, [{'word': 'two', 'start': 1, 'end': 2, 'score': .8}], .5)
    assert not w[0]['time_valid'] and w[1]['time_valid']
    assert low[0]['reason'] == 'alignment failed'


def test_numbers_contractions_hyphens_and_punctuation_preserved():
    words, units, issues = build_words("Don't buy 20 well-made items !")
    assert [w['text'] for w in words] == ["Don't", 'buy', '20', 'well-made', 'items', '!']
    assert [u['text'] for u in units if u['word_idx'] == 2] == ['twenty']
    assert [u['text'] for u in units if u['word_idx'] == 3] == ['well', 'made']
    assert not words[-1]['has_speech']
    assert issues


def test_degenerate_rows_not_fabricated_as_uniform():
    norm, zero = normalize_rows([[0, 0], [.2, .8]], 1e-12)
    assert zero.tolist() == [True, False]
    np.testing.assert_equal(norm[0], [0, 0])
    np.testing.assert_allclose(norm[1].sum(), 1)


def test_word_rows_average_and_compat_keeps_all_columns():
    r = np.array([[.1, .9], [.2, .8], [.4, .6], [.8, .2]], np.float32)
    rw = word_matrix(r, [-1, 0, 0, 1], 2)
    np.testing.assert_allclose(rw, [[.3, .7], [.8, .2]])
    full = pad(r, (8, 3))
    assert full[:2].shape == (2, 3)
    assert np.all(full[4:] == 0) and np.all(full[:, 2] == 0)


def test_trace_wrapper_returns_times_and_no_zero_importance_keyframes():
    sample = {'n_words': 1, 'words': [word('hi')], 'R_occ': np.ones((1, 1))}
    for m in ['audio', 'vision']:
        sample.update({m + '_pool_idx': [[0, 1]], m + '_frame_valid': [1, 0], m + '_frame_times': [.1, .2]})
    result = trace_sample(sample, [1], [1], [1])
    assert result['vision']['top_frames'] == [{'index': 0, 'time': .1, 'score': 1.0}]
    assert result['text']['unmapped_mass'] == 0


def test_negative_importance_and_invalid_frame_ranges_rejected():
    with pytest.raises(ValueError): text_position_to_words([-1], [[1]], 1)
    with pytest.raises(ValueError): modality_position_to_frames([1], [[0, 2]], [1], [0])


def test_gzip_and_plain_readers(tmp_path):
    for name, opener in [('a.pkl', open), ('b.pkl', gzip.open)]:
        p = tmp_path / name
        with opener(p, 'wb') as f: pickle.dump({'test': 1}, f)
        assert load_features(p) == {'test': 1}
