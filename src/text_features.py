from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

from .alignment import normalize_rows, word_matrix
from .common import record_model


def build_words(text, unicode_form='NFKC', overrides=None):
    """Whitespace words are the audit units; normalize ONLY the aligner's copy."""
    from num2words import num2words
    words, units, issues = [], [], []
    overrides = overrides or {}
    for match in re.finditer(r'\S+', text):
        original = match.group()
        norm = unicodedata.normalize(unicode_form, original).replace('’', "'").lower()
        speech = any(c.isalnum() for c in norm)
        word = {'text': original, 'norm': norm, 'char_start': match.start(),
                'char_end': match.end(), 'has_speech': speech, 'start': -1.0,
                'end': -1.0, 'score': None, 'time_valid': False}
        wi = len(words)
        if speech:
            if norm in overrides:
                aligned = overrides[norm]
                issues.append({'word': original, 'reason': 'configured normalization', 'rule': aligned})
            else:
                aligned = re.sub(r'\d+(?:\.\d+)?', lambda m: num2words(m.group()), norm)
                aligned = re.sub(r"[^a-z'\s]", ' ', aligned)
                aligned = re.sub(r'\s+', ' ', aligned).strip(" '")
                if re.search(r'\d', norm) or '-' in norm or '.' in norm:
                    issues.append({'word': original, 'reason': 'number/punctuation normalization; review abbreviations',
                                   'rule': aligned})
            if not aligned:
                issues.append({'word': original, 'reason': 'no English alignment units', 'rule': 'retain without time'})
            for unit in aligned.split():
                units.append({'text': unit, 'word_idx': wi})
        words.append(word)
    return words, units, issues


class BertEncoder:
    def __init__(self, ctx, load_weights=True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        cfg = ctx.cfg
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(cfg['torch_threads'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg['seed'])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device'] == 'auto' else cfg['device']
        self.tc = cfg['text']
        kwargs = {'revision': self.tc['revision'], 'local_files_only': self.tc['local_files_only']}
        self.tokenizer = AutoTokenizer.from_pretrained(self.tc['model'], use_fast=True, **kwargs)
        if not self.tokenizer.is_fast:
            raise ValueError('A fast tokenizer with word_ids() is required')
        if not load_weights:
            self.model = None
            return
        import requests
        for attempt in range(self.tc['download_retries']):
            try:
                self.model = AutoModel.from_pretrained(self.tc['model'], attn_implementation='eager', **kwargs)
                break
            except requests.RequestException:
                if attempt + 1 == self.tc['download_retries']:
                    raise
                ctx.logger('model_download').warning('Interrupted model transfer; retrying resumable cache (%d)', attempt + 1)
        self.model.to(device=self.device, dtype=torch.float32).eval()
        if self.tc['stable_linear']:
            import types
            # Algebraically identical Linear; each token uses the same 1xH GEMM
            # regardless of batch size. This avoids shape-dependent GEMM rounding
            # swamping small occlusion effects on some GPU/cuBLAS versions.
            def tokenwise_linear(module, x):
                shape = x.shape
                rows = x.reshape(-1, 1, shape[-1])
                weights = module.weight.t().unsqueeze(0).expand(len(rows), -1, -1)
                output = torch.bmm(rows, weights).reshape(*shape[:-1], module.out_features)
                return output + module.bias if module.bias is not None else output
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.forward = types.MethodType(tokenwise_linear, module)
        # Hash actual in-memory parameter bytes rather than only a model name.
        h = hashlib.sha256()
        for name, t in self.model.state_dict().items():
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        record_model(ctx, 'bert', {'name': self.tc['model'], 'revision': self.tc['revision'],
                     'resolved_commit': getattr(self.model.config, '_commit_hash', None),
                     'state_dict_sha256': h.hexdigest(), 'layer': self.tc['layer'],
                     'tokenizer_sha256': hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                     'dtype': 'float32', 'device': self.device, 'eval': True,
                     'stable_tokenwise_linear': self.tc['stable_linear'],
                     'inference_pad_multiple': self.tc['inference_pad_multiple']})

    def tokenize(self, words):
        tok = self.tokenizer([w['text'] for w in words], is_split_into_words=True,
                             add_special_tokens=True, truncation=False, return_attention_mask=True)
        length = len(tok['input_ids'])
        max_len = min(self.tc['max_positions'], self.model.config.max_position_embeddings if self.model is not None else self.tokenizer.model_max_length)
        if length > max_len:
            raise ValueError(f'{length} tokens exceeds {max_len}; never silently truncate')
        return {'input_ids': tok['input_ids'], 'attention_mask': tok['attention_mask'],
                'token_type_ids': tok.get('token_type_ids', [0] * length),
                'token_word_idx': [-1 if w is None else w for w in tok.word_ids()],
                'token_str': self.tokenizer.convert_ids_to_tokens(tok['input_ids'])}

    def tensors(self, unit):
        length = len(unit['input_ids'])
        multiple = self.tc.get('inference_pad_multiple', 1)
        width = min(((length + multiple - 1) // multiple) * multiple, self.model.config.max_position_embeddings)
        result = {}
        for k in ['input_ids', 'attention_mask', 'token_type_ids']:
            value = self.tokenizer.pad_token_id if k == 'input_ids' else 0
            padded = unit[k] + [value] * (width - length)
            result[k] = self.torch.tensor([padded], dtype=self.torch.long, device=self.device)
        return result

    def forward(self, inputs, attentions=False):
        out = self.model(**inputs, output_hidden_states=True, output_attentions=attentions)
        return out.hidden_states[self.tc['layer']], out.attentions

    def encode(self, unit):
        with self.torch.no_grad():
            h, _ = self.forward(self.tensors(unit))
        return h[0, :len(unit['input_ids'])].cpu().numpy().astype(np.float32)

    def batch_check(self, unit, extra):
        torch = self.torch
        inputs = self.tensors(unit)
        length = len(unit['input_ids'])
        # Both single and batch use the same configured padded width. Only real
        # positions are compared, and the other batch item has a different mask.
        batch = {k: t.repeat(2, 1) for k, t in inputs.items()}
        # Second sequence is genuinely shorter; first is the unmodified official sentence.
        if length > 3:
            other_length = max(3, length - extra)
            batch['input_ids'][1, other_length - 1] = self.tokenizer.sep_token_id
            batch['input_ids'][1, other_length:] = self.tokenizer.pad_token_id
            batch['attention_mask'][1, other_length:] = 0
        with torch.no_grad():
            single, _ = self.forward(inputs)
            batched, _ = self.forward(batch)
        return float((single[0, :length] - batched[0, :length]).abs().max().item())

    def occlusion(self, unit, n_words, epsilon):
        torch = self.torch
        inputs = self.tensors(unit)
        ids = np.asarray(unit['token_word_idx'])
        if n_words == 0:
            raise ValueError('No official words')
        batch = {k: v.repeat(n_words, 1) for k, v in inputs.items()}
        for j in range(n_words):
            rows = np.flatnonzero(ids == j)
            if not len(rows):
                raise ValueError(f'Official word {j} has no tokens')
            batch['input_ids'][j, torch.as_tensor(rows, device=self.device)] = self.tokenizer.mask_token_id
        # Exactly W variants in one batch, as requested. OOM is logged, never silently changes method.
        with torch.no_grad():
            original, _ = self.forward(inputs)
            masked, _ = self.forward(batch)
            raw = (1 - torch.nn.functional.cosine_similarity(original, masked, dim=-1)).clamp_min(0)
        raw = raw[:, :len(unit['input_ids'])].T.cpu().numpy().astype(np.float32)
        norm, zero = normalize_rows(raw, epsilon)
        return {'R_raw': raw, 'R_occ': norm, 'R_word': word_matrix(norm, ids, n_words),
                'degenerate_rows': np.flatnonzero(zero).astype(np.int32)}

    def rollout(self, unit, n_words, epsilon):
        with self.torch.no_grad():
            _, att = self.forward(self.tensors(unit), attentions=True)
        L = len(unit['input_ids'])
        joint = np.eye(L, dtype=np.float32)
        layer = self.tc['layer']
        depth = len(att) if layer == -1 else (layer if layer >= 0 else len(att) + 1 + layer)
        for a in att[:depth]:
            a = a[0, :, :L, :L].mean(dim=0).cpu().numpy().astype(np.float32) + np.eye(L, dtype=np.float32)
            a /= a.sum(axis=1, keepdims=True)
            joint = a @ joint
        r = np.zeros((L, n_words), np.float32)
        ids = np.asarray(unit['token_word_idx'])
        for j in range(n_words):
            r[:, j] = joint[:, ids == j].sum(axis=1)
        # Special-token mass is excluded then renormalized; not a causal contribution.
        return normalize_rows(r, epsilon)[0]


def match_alignment(words, units, result_words, min_score):
    """Monotonic sequence matching; no fabricated times for unmatched units."""
    from difflib import SequenceMatcher
    def canon(s):
        return re.sub(r"[^a-z0-9']", '', s.lower().replace('’', "'"))
    matcher = SequenceMatcher(a=[canon(x['text']) for x in units],
                              b=[canon(x.get('word', x.get('text', ''))) for x in result_words],
                              autojunk=False)
    mapped = {}
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            mapped[block.a + delta] = result_words[block.b + delta]
    low, audit = [], []
    for j, w in enumerate(words):
        positions = [i for i, u in enumerate(units) if u['word_idx'] == j]
        resolved = [mapped.get(i, {}) for i in positions]
        good = bool(positions) and all(x.get('start') is not None and x.get('end') is not None
                                      and x['end'] > x['start'] for x in resolved)
        if good:
            w['start'] = float(min(x['start'] for x in resolved))
            w['end'] = float(max(x['end'] for x in resolved))
            scores = [float(x['score']) for x in resolved if x.get('score') is not None]
            w['score'] = min(scores) if len(scores) == len(resolved) else None
            w['time_valid'] = True
            if w['score'] is None or w['score'] < min_score:
                low.append({'word_idx': j, 'word': w['text'], 'score': w['score'],
                            'reason': 'unknown confidence' if w['score'] is None else 'low confidence'})
        elif w['has_speech']:
            low.append({'word_idx': j, 'word': w['text'], 'score': None, 'reason': 'alignment failed'})
        for i in positions:
            audit.append({**units[i], 'alignment_unit_idx': i, 'matched': i in mapped,
                          'result': mapped.get(i)})
    return words, low, audit
