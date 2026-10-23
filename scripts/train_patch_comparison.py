"""Launch isolated five-fold patch experiments using the existing trainer.

Without --run or --smoke, print commands only. No dependencies are installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as settings


def commands(patches, root: Path, *, smoke: bool, resume: bool):
    trainer = settings.PROJECT_ROOT / 'scripts/train.py'
    splits = root / 'splits.json'
    base = [sys.executable, '-u', str(trainer)]
    # --prepare audits the cohort and creates deterministic patient splits.
    # Both patch sizes use the same immutable manifest, including inner validation.
    if not splits.is_file():
        yield base + ['--prepare', '--output', str(root / 'preparation'), '--splits', str(splits)]
    for patch in patches:
        output = root / (f'smoke{patch}' if smoke else f'patch{patch}')
        mode = ['--smoke'] if smoke else ['--fold', 'all']
        command = base + mode + [
            '--patch', str(patch), '--output', str(output), '--splits', str(splits),
            '--sw-batch-size', str(settings.PATCH_COMPARISON_SW_BATCH_SIZE),
        ]
        if resume:
            command.append('--resume')
        yield command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Train all five folds for each selected patch size')
    mode.add_argument('--smoke', action='store_true', help='Bounded fit check for each size; no held-out predictions')
    parser.add_argument('--patch', type=int, choices=settings.PATCH_COMPARISON_SIZES,
                        help='Run only this patch size; default: both, sequentially')
    parser.add_argument('--resume', action='store_true', help='Resume full runs; verify/skip completed folds')
    parser.add_argument('--output-root', type=Path, default=settings.PATCH_COMPARISON_OUTPUT_ROOT,
                        help='Separate experiment root; keep unchanged on resume')
    args = parser.parse_args(argv)
    if args.resume and args.smoke:
        parser.error('--resume is for full runs; use a fresh --output-root for repeated smoke checks')
    root = (settings.PROJECT_ROOT / args.output_root).resolve()
    if root == settings.OUTPUT_ROOT.resolve() or root == settings.DATA_ROOT.resolve() or settings.DATA_ROOT.resolve() in root.parents:
        parser.error('Choose a separate comparison root outside the source data and original experiment')
    patches = [args.patch] if args.patch is not None else settings.PATCH_COMPARISON_SIZES
    execute = args.run or args.smoke
    print('Running sequentially.' if execute else 'Preview only; add --smoke or --run to execute.', flush=True)
    for command in commands(patches, root, smoke=args.smoke, resume=args.resume):
        print(shlex.join(command), flush=True)
        if execute:
            subprocess.run(command, cwd=settings.PROJECT_ROOT, check=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Resume full runs with --run --resume; only completed epochs are saved.', file=sys.stderr)
        sys.exit(130)
    except subprocess.CalledProcessError as error:
        print(f'Child command failed ({error.returncode}); later runs were not started. Inspect its log.', file=sys.stderr)
        sys.exit(error.returncode if error.returncode > 0 else 1)
