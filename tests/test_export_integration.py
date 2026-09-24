"""Synthetic arrays exercise serialization and truncation, never real feature outputs."""
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import Context, write_json
from src.export import export, load_features, verify_export


def test_full_compat_retains_context_columns_and_padding(tmp_path):
    base = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((base / 'config.yaml').read_text(encoding='utf-8'))
    cfg['output_dir'] = 'out'; cfg['data_root'] = '.'
    cfg['export']['target_mb'] = 25
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    ctx = Context(path)
    sid = 'fixture$_$0'
    L, W = 54, 52
    words = [{'text': f'w{i}', 'norm': f'w{i}', 'start': i, 'end': i + 1,
              'score': 1, 'has_speech': True, 'time_valid': True} for i in range(W)]
    word_ids = np.r_[-1, np.arange(W), -1].astype(np.int32)
    masks = np.r_[0, np.ones(W), 0].astype(np.int32)
    index = np.full((L, 2), -1, np.int32); index[1:-1] = np.arange(W)[:, None]
    a = {'text': np.ones((L, 4), np.float32), 'audio': masks[:, None] * np.ones((L, 2), np.float32),
         'vision': masks[:, None] * np.ones((L, 3), np.float32), 'token_word_idx': word_ids,
         'text_valid_mask': np.ones(L, np.int32), 'audio_valid_mask': masks, 'vision_valid_mask': masks,
         'invalid_reason_audio': 1-masks, 'invalid_reason_vision': 1-masks,
         'token_time': np.stack([word_ids, word_ids+1],axis=1).astype(np.float32),
         'audio_pool_idx': index, 'vision_pool_idx': index}
    a['token_time'][[0, -1]] = -1
    np.savez_compressed(ctx.sample_path('aligned', sid, 'npz'), **a)
    weights = [{'indices': [i-1], 'weights': [1.]} if 0 < i < L-1 else {'indices': [], 'weights': []} for i in range(L)]
    write_json(ctx.sample_path('aligned', sid), {'words': words, 'token_str': ['[CLS]'] + [w['text'] for w in words] + ['[SEP]'],
               'pool_weights': {'audio': weights, 'vision': weights}, 'quality_flags': {}})
    write_json(ctx.sample_path('text_units', sid), {'words': words, 'input_ids': list(range(L))})
    R = np.full((L, W), 1/W, np.float32)
    np.savez_compressed(ctx.sample_path('receptive', sid, 'npz'), R_occ=R, R_raw=R*.1,
                        R_word=np.full((W,W), 1/W, np.float32), R_rollout=R)
    for m, D in [('audio', 2), ('vision', 3)]:
        np.savez_compressed(ctx.sample_path(m, sid, 'npz'), features=np.ones((W,D), np.float32),
                            times=np.arange(W,dtype=np.float32), intervals=np.stack([np.arange(W),np.arange(W)+1],axis=1),valid=np.ones(W,np.int32))
        write_json(ctx.sample_path(m, sid), {'dimension': D, 'context': 'synthetic fixture'})
    write_json(ctx.path('meta', 'manifest.json'), [{'sample_id':sid, 'video_path':'fixture_only'}])
    write_json(ctx.path('meta', 'length_policy.json'), {'L_max_pad':56,'W_max':W,'needs_compat':True})
    write_json(ctx.path('meta', 'quality_report.json'), {'all_passed':True,'samples':[{'sample_id':sid,'passed':True}]})
    export(ctx)
    full = load_features(ctx.path('features','q1_features_full.pkl'))
    compat = load_features(ctx.path('features','q1_features_compat50.pkl'))
    assert full['text'].shape == (1,56,4)
    assert compat['text'].shape == (1,50,4)
    assert compat['R_occ'].shape == (1,50,W)
    assert compat['R_word'].shape == (1,W,W)
    assert compat['valid_length'][0] == 50
    assert compat['valid_length_full'][0] == 54
    assert full['invalid_reason_audio'][0,-1] == 2
    assert full['token_word_idx'][0,-1] == -1
    assert not verify_export(full,.001)
    np.testing.assert_equal(compat['R_occ'][0,:,51],full['R_occ'][0,:50,51])
