"""Offline test of actual BERT masking mechanics using random tiny weights.

These are test fixtures only. They are never exported as dataset features.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from src.text_features import BertEncoder


@pytest.fixture
def encoder():
    torch.manual_seed(7)
    e = BertEncoder.__new__(BertEncoder)
    e.torch = torch
    e.device = 'cpu'
    e.tc = {'layer': -1}
    e.tokenizer = SimpleNamespace(mask_token_id=4, pad_token_id=0, sep_token_id=2)
    cfg = transformers.BertConfig(vocab_size=32, hidden_size=12, num_hidden_layers=2,
                                  num_attention_heads=3, intermediate_size=24,
                                  hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    cfg._attn_implementation = 'eager'
    e.model = transformers.BertModel(cfg).eval()
    return e


def unit():
    return {'input_ids': [1, 5, 6, 7, 2], 'attention_mask': [1]*5,
            'token_type_ids': [0]*5, 'token_word_idx': [-1, 0, 0, 1, -1]}


def test_mask_all_subwords_in_one_batch_and_keep_special_rows(encoder):
    captured = []
    real_forward = encoder.forward
    def forward(inputs, attentions=False):
        captured.append(inputs['input_ids'].detach().clone())
        return real_forward(inputs, attentions)
    encoder.forward = forward
    result = encoder.occlusion(unit(), 2, 1e-12)
    assert captured[1].shape == (2, 5)
    assert captured[1].tolist() == [[1, 4, 4, 7, 2], [1, 5, 6, 4, 2]]
    assert result['R_occ'].shape == (5, 2)
    assert result['R_word'].shape == (2, 2)
    np.testing.assert_allclose(result['R_occ'].sum(axis=1), 1, atol=1e-5)
    again = encoder.occlusion(unit(), 2, 1e-12)
    np.testing.assert_equal(result['R_occ'], again['R_occ'])


def test_padding_attention_and_rollout(encoder):
    assert encoder.batch_check(unit(), 3) < 1e-5
    rollout = encoder.rollout(unit(), 2, 1e-12)
    assert rollout.shape == (5, 2)
    np.testing.assert_allclose(rollout.sum(axis=1), 1, atol=1e-6)
