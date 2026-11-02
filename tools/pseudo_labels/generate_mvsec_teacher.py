#!/usr/bin/env python3
"""Generate teacher predictions from prepared MVSEC RGB frames.

The optional GT inverse-affine fit follows the original MVSEC preparation code.
Run align_mvsec_quantiles.py on the fitted depth maps to generate pseudo-depth.
"""
import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image


def sequence_root(root, sequence):
    prefix = 'mvsec_dataset_day2_'
    if sequence.startswith(prefix):
        return root / 'mvsec_dataset_day2' / sequence[len(prefix):]
    return root / sequence


def fit_inverse_affine(prediction, gt):
    """Fit 1/GT = scale * relative_prediction + shift, then invert."""
    prediction = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if prediction.shape != gt.shape:
        raise ValueError(f'RGB prediction and GT dimensions differ: {prediction.shape} and {gt.shape}')
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(prediction)
    if valid.sum() < 2:
        raise ValueError('at least two valid GT pixels are required for the inverse affine fit')
    values = prediction[valid]
    matrix = np.vstack([values, np.ones_like(values)]).T
    if np.ptp(values) == 0:
        raise ValueError('teacher prediction is constant at valid GT pixels')
    scale, shift = np.linalg.lstsq(matrix, 1.0 / np.clip(gt[valid], 1e-8, None), rcond=None)[0]
    return (1.0 / np.clip(scale * prediction + shift, 1e-8, None)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mvsec-root', type=Path, required=True,
                        help='Prepared MVSEC root containing sequence directories')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--sequence', action='append', required=True,
                        help='Sequence name, repeat for more sequences')
    parser.add_argument('--model', default='depth-anything/Depth-Anything-V2-Large-hf',
                        help='Hugging Face model ID or local model directory')
    parser.add_argument('--inverse-affine-gt-root', type=Path,
                        help='Fit teacher to inverse metric GT before saving; root has same layout as MVSEC root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--frame-index', type=int, action='append', help='Exact RGB frame IDs; repeat')
    parser.add_argument('--max-frames', type=int, default=0, help='Global limit; 0 means all')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error('--max-frames must be non-negative')
    if args.output_root.resolve() == args.mvsec_root.resolve():
        parser.error('--output-root must differ from --mvsec-root')
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.config.depth_estimation_type = 'relative'
    model = model.to(args.device).eval()
    pattern = re.compile(r'^frame_(\d+)\.png$')
    wanted = set(args.frame_index) if args.frame_index else None
    generated = skipped = seen = 0
    for sequence in args.sequence:
        if Path(sequence).name != sequence:
            parser.error('--sequence must be a single directory name')
        folder = 'davis' if sequence.startswith('mvsec_dataset_day2_') else 'davis_left_sync'
        rgb_dir = sequence_root(args.mvsec_root, sequence) / 'rgb' / folder
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f'missing prepared RGB directory: {rgb_dir}')
        paths = sorted(p for p in rgb_dir.iterdir() if p.is_file() and pattern.fullmatch(p.name))
        if wanted:
            paths = [p for p in paths if int(pattern.fullmatch(p.name).group(1)) in wanted]
            found = {int(pattern.fullmatch(p.name).group(1)) for p in paths}
            if found != wanted:
                raise FileNotFoundError(f'missing requested RGB frame(s) for {sequence}: {sorted(wanted - found)}')
        if not paths:
            raise FileNotFoundError(f'no RGB frames in {rgb_dir}')
        for rgb_path in paths:
            if args.max_frames and seen >= args.max_frames:
                print(f'Generated {generated}; skipped {skipped}.')
                return
            seen += 1
            frame = int(pattern.fullmatch(rgb_path.name).group(1))
            output = args.output_root / sequence / f'frame_{frame:010d}_pred_depth.npy'
            if output.exists() and not args.overwrite:
                skipped += 1
                continue
            with Image.open(rgb_path) as src:
                image = src.convert('RGB')
            width, height = image.size
            inputs = {k: v.to(args.device) for k, v in processor(images=image, return_tensors='pt').items()}
            with torch.no_grad():
                prediction = model(**inputs)
            if hasattr(processor, 'post_process_depth_estimation'):
                depth = processor.post_process_depth_estimation(
                    prediction, target_sizes=[(height, width)])[0]['predicted_depth']
            else:
                # Transformers 4.41 uses DPTImageProcessor without this method.
                # Match the original project's compatibility fallback.
                depth = prediction.predicted_depth
                if depth.ndim == 3:
                    depth = depth.unsqueeze(1)
                depth = torch.nn.functional.interpolate(
                    depth, size=(height, width), mode='bilinear', align_corners=False)
            depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(depth).all():
                raise ValueError(f'non-finite teacher prediction: {rgb_path}')
            if args.inverse_affine_gt_root is not None:
                gt_path = sequence_root(args.inverse_affine_gt_root, sequence) / 'depth' / 'data' / f'depth_{frame:010d}.npy'
                depth = fit_inverse_affine(depth, np.load(gt_path, allow_pickle=False))
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('wb' if args.overwrite else 'xb') as handle:
                np.save(handle, depth, allow_pickle=False)
            generated += 1
            if generated == 1 or generated % 100 == 0:
                print(f'{sequence} frame {frame}: generated={generated}', flush=True)
    print(f'Generated {generated}; skipped {skipped}.')


if __name__ == '__main__':
    main()
