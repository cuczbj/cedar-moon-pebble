from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from filelock import FileLock

import numpy as np
import yaml


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable,
                              allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, default=jsonable)
                        if isinstance(v, (list, dict)) else v for k, v in r.items()})


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_command(args, cwd=None):
    p = subprocess.run([str(x) for x in args], cwd=cwd, text=True,
                       encoding='utf-8', errors='replace', capture_output=True)
    if p.returncode:
        raise RuntimeError(f'Command failed ({p.returncode}): {args}\n{p.stderr[-6000:]}\n{p.stdout[-3000:]}')
    return p.stdout


class Context:
    def __init__(self, config, limit=None):
        self.config_path = Path(config).resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding='utf-8'))
        self.base = self.config_path.parent
        self.root = (self.base / self.cfg['data_root']).resolve()
        self.out = (self.base / self.cfg['output_dir']).resolve()
        if limit:
            self.out = self.out.with_name(self.out.name + f'_smoke_{limit}')
        self.limit = limit
        cache = (self.base / self.cfg['model_cache']).resolve()
        os.environ['HF_HOME'] = str(cache / 'huggingface')
        os.environ['TORCH_HOME'] = str(cache / 'torch')
        os.environ['NLTK_DATA'] = str(cache / 'nltk')
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        os.environ['MPLCONFIGDIR'] = str(self.out / 'intermediate' / 'matplotlib_cache')
        for sub in ['features', 'meta', 'logs', 'figures', 'intermediate']:
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        random.seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])

    def path(self, *parts):
        return self.out.joinpath(*parts)

    def sample_path(self, stage, sid, ext='json'):
        p = self.path('intermediate', stage, f'{sid}.{ext}')
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def manifest(self):
        return read_json(self.path('meta', 'manifest.json'))

    def logger(self, step):
        logger = logging.getLogger(step)
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        for h in [logging.FileHandler(self.path('logs', step + '.log'), encoding='utf-8'),
                  logging.StreamHandler(sys.stdout)]:
            h.setFormatter(fmt)
            logger.addHandler(h)
        return logger

    def each(self, step, fn):
        log = self.logger(step)
        start = time.monotonic()
        records = []
        for row in self.manifest():
            sid = row['sample_id']
            try:
                fn(row)
                records.append({'sample_id': sid, 'status': 'ok', 'reason': ''})
                log.info('%s ok', sid)
            except Exception as e:
                records.append({'sample_id': sid, 'status': 'failed', 'reason': repr(e)})
                log.exception('%s failed', sid)
        write_json(self.path('logs', step + '_status.json'), records)
        log.info('processed=%s failed=%s elapsed_s=%.2f', len(records),
                 sum(r['status'] != 'ok' for r in records), time.monotonic() - start)
        return records


def record_environment(ctx):
    packages = {}
    for p in ['torch', 'torchaudio', 'transformers', 'whisperx', 'opensmile', 'av',
              'numpy', 'openpyxl', 'num2words', 'praatio', 'PyYAML']:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    ffmpeg = None
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg = {'executable': exe, 'version': run_command([exe, '-version']).splitlines()[0]}
    except Exception as e:
        ffmpeg = {'error': str(e)}
    of = (ctx.base / ctx.cfg['vision']['executable']).resolve()
    result = {'python': sys.version, 'executable': sys.executable, 'packages': packages,
              'ffmpeg': ffmpeg, 'config_sha256': sha256(ctx.config_path), 'config': ctx.cfg,
              'openface': {'path': str(of), 'exists': of.is_file(),
                           'version_label': ctx.cfg['vision']['version_label']},
              'models': {}, 'smoke_limit': ctx.limit}
    result['source_hashes'] = {str(p.relative_to(ctx.base)): sha256(p)
                               for p in (ctx.base / 'src').glob('*.py')}
    result['source_hashes']['run.py'] = sha256(ctx.base / 'run.py')
    if of.is_file():
        result['openface']['sha256'] = sha256(of)
        result['openface']['model_hashes'] = {
            str(p.relative_to(of.parent)): sha256(p)
            for folder in ('model', 'AU_predictors')
            for p in (of.parent / folder).rglob('*') if p.is_file()
        }
    old = ctx.path('logs', 'environment.json')
    with FileLock(str(old) + '.lock'):
        if old.exists():
            result['models'] = read_json(old).get('models', {})
        write_json(old, result)


def record_model(ctx, name, info):
    p = ctx.path('logs', 'environment.json')
    with FileLock(str(p) + '.lock'):
        e = read_json(p) if p.exists() else {'models': {}}
        e.setdefault('models', {})[name] = info
        write_json(p, e)
