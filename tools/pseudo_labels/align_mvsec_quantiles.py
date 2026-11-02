#!/usr/bin/env python3
"""Align prepared MVSEC teacher depth maps to GT by depth-space quantiles.

This is the alignment step only: RGB/event registration, teacher inference,
and preparation of GT arrays must already have been performed.
"""
import argparse
from pathlib import Path
import re

import numpy as np


def align_quantiles(teacher, gt, num_quantiles=64, quantile_min=0.05,
                    quantile_max=0.95, min_valid=1e-3, max_depth=80.0,
                    min_points=32):
    """Match teacher and GT depth distributions at jointly valid pixels."""
    teacher = np.asarray(teacher, dtype=np.float32).squeeze()
    gt = np.asarray(gt, dtype=np.float32).squeeze()
    if teacher.ndim != 2 or gt.ndim != 2:
        raise ValueError('teacher and GT must each be 2-D depth arrays')
    # The prepared labels and GT loaders crop bottom/right to multiples of 8.
    teacher = teacher[:teacher.shape[0] // 8 * 8, :teacher.shape[1] // 8 * 8]
    gt = gt[:gt.shape[0] // 8 * 8, :gt.shape[1] // 8 * 8]
    if teacher.shape != gt.shape or not teacher.size:
        raise ValueError(f'incompatible cropped teacher/GT shapes: {teacher.shape}, {gt.shape}')
    if not np.isfinite(teacher).all():
        raise ValueError('teacher prediction must be finite over the dense output')
    if num_quantiles < 2 or not 0 <= quantile_min < quantile_max <= 1:
        raise ValueError('require at least two quantiles and 0 <= min < max <= 1')
    if not 0 < min_valid < max_depth or min_points < 1:
        raise ValueError('require 0 < min_valid < max_depth and min_points >= 1')
    teacher = np.clip(teacher, 0.0, max_depth)
    valid = np.isfinite(gt) & (teacher > min_valid) & (gt > min_valid) & (gt < max_depth)
    count = int(valid.sum())
    if count < min_points:
        raise ValueError(f'need at least {min_points} valid pixels, got {count}')
    levels = np.linspace(quantile_min, quantile_max, num_quantiles)
    source = np.quantile(teacher[valid], levels)
    target = np.quantile(gt[valid], levels)
    source, unique_indices = np.unique(source, return_index=True)
    target = target[unique_indices]
    if source.size < 2:
        raise ValueError('teacher has fewer than two distinct quantile knots')
    aligned = np.interp(teacher.astype(np.float64), source, target,
                        left=float(target[0]), right=float(target[-1]))
    return np.clip(aligned, 0.0, max_depth).astype(np.float32), count


def gt_sequence_name(sequence):
    prefix = 'mvsec_dataset_day2_'
    if sequence.startswith(prefix):
        return Path('mvsec_dataset_day2') / sequence[len(prefix):]
    return Path(sequence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-root', required=True, type=Path)
    parser.add_argument('--gt-root', required=True, type=Path,
                        help='MVSEC GT train root containing sequence/depth/data')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--sequence', required=True, action='append', help='Repeat for each sequence')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit total frames; 0 means all')
    parser.add_argument('--num-quantiles', type=int, default=64)
    parser.add_argument('--quantile-min', type=float, default=0.05)
    parser.add_argument('--quantile-max', type=float, default=0.95)
    parser.add_argument('--min-valid', type=float, default=1e-3)
    parser.add_argument('--max-depth', type=float, default=80.0)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output labels')
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error('--max-frames must be non-negative')
    if args.output_root.resolve() in {args.teacher_root.resolve(), args.gt_root.resolve()}:
        parser.error('--output-root must differ from the input roots')
    pattern = re.compile(r'^frame_(\d+)_pred_depth\.npy$')
    generated = skipped = seen = 0
    for sequence in args.sequence:
        if Path(sequence).name != sequence:
            parser.error('--sequence must be a single directory name')
        input_dir = args.teacher_root / sequence
        if not input_dir.is_dir():
            raise FileNotFoundError(f'missing teacher sequence: {input_dir}')
        paths = sorted(p for p in input_dir.iterdir() if p.is_file() and pattern.fullmatch(p.name))
        if not paths:
            raise FileNotFoundError(f'no teacher depth maps found in {input_dir}')
        for prediction_path in paths:
            if args.max_frames and seen >= args.max_frames:
                print(f'Generated {generated}, skipped {skipped}.')
                return
            seen += 1
            frame = int(pattern.fullmatch(prediction_path.name).group(1))
            gt_path = args.gt_root / gt_sequence_name(sequence) / 'depth/data' / f'depth_{frame:010d}.npy'
            output = args.output_root / sequence / prediction_path.name
            if output.exists() and not args.overwrite:
                skipped += 1
                continue
            if not gt_path.is_file():
                raise FileNotFoundError(f'missing GT for {prediction_path}: {gt_path}')
            aligned, count = align_quantiles(np.load(prediction_path, allow_pickle=False),
                                             np.load(gt_path, allow_pickle=False),
                                             args.num_quantiles, args.quantile_min,
                                             args.quantile_max, args.min_valid, args.max_depth)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation avoids overwriting another run unless explicitly requested.
            with output.open('wb' if args.overwrite else 'xb') as handle:
                np.save(handle, aligned, allow_pickle=False)
            generated += 1
            if generated == 1 or generated % 100 == 0:
                print(f'{sequence} frame {frame}: valid={count}, generated={generated}', flush=True)
    print(f'Generated {generated}, skipped {skipped}.')


if __name__ == '__main__':
    main()
