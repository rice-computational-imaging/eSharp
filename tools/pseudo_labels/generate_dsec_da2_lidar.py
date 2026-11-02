#!/usr/bin/env python3
"""Generate DSEC DA2/LiDAR pseudo-depth using depth-aware two-view alignment.

Requires DSEC RGB, event disparity and calibration files, plus a separate
Depth-Anything-V2 source checkout and ViT-L checkpoint. See PSEUDO_LABELS.md.
"""
import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from dsec_alignment.alignment import robust_scale_shift
from dsec_alignment.calibration import load_dsec_calibration
from dsec_alignment.geometry import remap_frame_to_event, reproject_event_depth_to_frame


def bounded_da2_depth(relative, lidar_depth, valid, min_depth, max_depth):
    """Fit relative DA2 output to inverse LiDAR depth, then invert and bound."""
    inverse_lidar = np.full(lidar_depth.shape, np.nan, dtype=np.float32)
    lidar_valid = valid & np.isfinite(lidar_depth) & (lidar_depth > 0)
    inverse_lidar[lidar_valid] = 1.0 / lidar_depth[lidar_valid]
    result = robust_scale_shift(relative, inverse_lidar, lidar_valid,
                                min_depth=np.finfo(np.float32).eps)
    inverse_depth = np.clip(result.depth, 1.0 / max_depth, 1.0 / min_depth)
    return np.clip(1.0 / inverse_depth, min_depth, max_depth).astype(np.float32)


def disparity_to_depth(disparity, focal_px, baseline_m):
    disparity = np.asarray(disparity, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = focal_px * baseline_m / disparity
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth.astype(np.float32)


def load_teacher(repo, checkpoint, device):
    import torch
    if not (repo / 'depth_anything_v2/dpt.py').is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Provide a Depth-Anything-V2 checkout and its ViT-L checkpoint')
    sys.path.insert(0, str(repo.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return model.to(device).eval()


def generate_frame(root, sequence, frame_id, model, min_depth, max_depth):
    import torch
    calibration = load_dsec_calibration(root / 'calib' / sequence / 'calibration/cam_to_cam.yaml')
    rgb_path = root / 'rgb_frames' / sequence / 'images/left/rectified' / f'{frame_id:06d}.png'
    disp_path = root / 'disparity_maps' / sequence / 'disparity/event' / f'{frame_id:06d}.png'
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    raw = np.asarray(Image.open(disp_path), dtype=np.float32)
    disparity = np.where(raw > 0, raw / 256.0, np.nan)
    lidar_event = disparity_to_depth(disparity, calibration.K_event[0, 0], calibration.event_stereo_baseline_m)
    lidar_frame, frame_valid = reproject_event_depth_to_frame(
        lidar_event, calibration.K_event, calibration.K_frame,
        calibration.event_to_frame, calibration.frame_size)
    with torch.inference_mode():
        relative_frame = model.infer_image(rgb[..., ::-1], 518).astype(np.float32)
    frame_depth = bounded_da2_depth(relative_frame, lidar_frame, frame_valid, min_depth, max_depth)
    event_rgb = remap_frame_to_event(
        rgb, frame_depth, calibration.K_frame, calibration.K_event,
        calibration.frame_to_event, calibration.event_size, fill_holes=True).image_filled
    with torch.inference_mode():
        relative_event = model.infer_image(event_rgb[..., ::-1], 518).astype(np.float32)
    event_valid = np.isfinite(lidar_event) & (lidar_event > 0)
    depth = bounded_da2_depth(relative_event, lidar_event, event_valid, min_depth, max_depth)
    if not np.isfinite(depth).all():
        raise ValueError('non-finite pseudo-depth')
    return depth


def save_output(path, depth, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pseudo-', suffix='.npy', delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, depth, allow_pickle=False)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic creation; refuses to overwrite.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsec-root', type=Path, required=True)
    parser.add_argument('--da2-repo', type=Path, required=True)
    parser.add_argument('--da2-checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--sequence', action='append', required=True, help='Repeat; or specify all')
    parser.add_argument('--frame-index', action='append', type=int, help='Optional exact disparity frame ID; repeat')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum frames globally, 0 means all')
    parser.add_argument('--min-depth', type=float, default=0.5)
    parser.add_argument('--max-depth', type=float, default=80.0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--device', default='cuda:0', help='Device after CUDA_VISIBLE_DEVICES filtering; cpu supported')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output labels')
    parser.add_argument('--dry-run', action='store_true', help='List selected work without loading the model or writing')
    args = parser.parse_args()
    if not 0 < args.min_depth < args.max_depth or args.max_frames < 0:
        parser.error('require 0 < min-depth < max-depth and non-negative max-frames')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('require num-shards > 0 and 0 <= shard-index < num-shards')
    if args.output_root.resolve() == args.dsec_root.resolve():
        parser.error('output root must differ from DSEC input root')
    disp_root = args.dsec_root / 'disparity_maps'
    sequences = sorted(p.name for p in disp_root.iterdir() if p.is_dir()) if args.sequence == ['all'] else args.sequence
    if 'all' in sequences or any(Path(s).name != s for s in sequences):
        parser.error('use sequence names, or pass --sequence all by itself')
    work = []
    wanted = set(args.frame_index) if args.frame_index else None
    for sequence in sequences:
        disparity_dir = disp_root / sequence / 'disparity/event'
        rgb_dir = args.dsec_root / 'rgb_frames' / sequence / 'images/left/rectified'
        if not disparity_dir.is_dir() or not rgb_dir.is_dir():
            raise FileNotFoundError(f'missing disparity or RGB directory for {sequence}')
        for path in sorted(disparity_dir.glob('*.png')):
            if not path.stem.isdecimal():
                continue
            frame = int(path.stem)
            if (wanted is None or frame in wanted) and (rgb_dir / f'{frame:06d}.png').is_file():
                work.append((sequence, frame))
    if wanted and {frame for _, frame in work} != wanted:
        raise FileNotFoundError('one or more requested frame indices lack matching RGB/disparity')
    if args.max_frames:
        work = work[:args.max_frames]
    work = work[args.shard_index::args.num_shards]
    if args.dry_run:
        print(f'Selected {len(work)} frames; first: {work[:5]}')
        return
    if not work:
        raise ValueError('no frames selected')
    import torch
    torch.set_num_threads(args.cpu_threads)
    model = load_teacher(args.da2_repo, args.da2_checkpoint, args.device)
    generated = skipped = 0
    for sequence, frame in work:
        output = args.output_root / sequence / f'frame{frame}_pred_depth.npy'
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        depth = generate_frame(args.dsec_root, sequence, frame, model, args.min_depth, args.max_depth)
        save_output(output, depth, args.overwrite)
        generated += 1
        if generated == 1 or generated % 25 == 0:
            print(f'{sequence} frame {frame}: generated={generated} skipped={skipped}', flush=True)
    print(f'Generated {generated}; skipped {skipped}.')


if __name__ == '__main__':
    main()
