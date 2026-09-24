from pathlib import Path
import argparse
import sys

from src.common import Context, record_environment, write_json
from src import steps

STEPS = ['s01_manifest', 's02_probe_media', 's03_preprocess_av', 's04_text_units',
         's05_force_align', 's06_text_encode', 's06b_text_receptive_field', 's07_audio_features',
         's08_visual_features', 's09_11_align_pool_mask', 's12_length_policy',
         's13_14_quality_check', 's15_export', 's16_visualize']


def main(default_step=None):
    parser = argparse.ArgumentParser(description='Q1 official-text wordpiece feature pipeline')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--step', choices=STEPS + ['all'], default=default_step or 'all')
    parser.add_argument('--limit', type=int, help='Smoke run in a separate output directory; never a final submission')
    parser.add_argument('--from-step', choices=STEPS)
    parser.add_argument('--to-step', choices=STEPS)
    args = parser.parse_args()
    ctx = Context(args.config, args.limit)
    record_environment(ctx)
    selected = list(STEPS) if args.step == 'all' else [args.step]
    if args.from_step:
        selected = selected[selected.index(args.from_step):]
    if args.to_step:
        selected = selected[:selected.index(args.to_step) + 1]
    failures = []
    for name in selected:
        try:
            getattr(steps, name)(ctx)
            status = ctx.path('logs', name + '_status.json')
            if status.exists():
                from src.common import read_json
                failed = [r for r in read_json(status) if r['status'] == 'failed']
                if failed:
                    failures.append({'step': name, 'failed_samples': failed})
        except Exception as e:
            ctx.logger(name).exception('Stage failed')
            failures.append({'step': name, 'error': repr(e)})
    write_json(ctx.path('logs', 'last_run.json'), {'steps': selected, 'failures': failures})
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
