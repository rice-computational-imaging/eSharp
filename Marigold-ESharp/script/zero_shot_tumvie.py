#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters used by event datasets
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from marigold.marigold_pipeline import MarigoldPipeline
from src.trainer import get_trainer_cls
from src.util.depth_transform import get_depth_normalizer
from src.util.logging_util import inverse_log_magma_numpy, save_rgb_svg

DEFAULT_STAGE1 = Path('/path/to/repository/Marigold/output/train_marigold_dsec_old_nbin_/checkpoint/latest')
DEFAULT_STAGE2 = Path('/path/to/repository/Marigold/output/train_marigold_dsec_old_nbin_calib_joint_/checkpoint/latest')


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _collect_h5_datasets(group, prefix='') -> dict[str, h5py.Dataset]:
    out = {}
    for k, v in group.items():
        name = f'{prefix}/{k}' if prefix else k
        if isinstance(v, h5py.Dataset):
            out[name] = v
        elif isinstance(v, h5py.Group):
            out.update(_collect_h5_datasets(v, name))
    return out


def _event_datasets(f: h5py.File):
    ds = _collect_h5_datasets(f)
    names = set(ds)
    if {'events/x', 'events/y', 'events/t', 'events/p'}.issubset(names):
        return ds['events/x'], ds['events/y'], ds['events/t'], ds['events/p']
    if 'events' in ds and ds['events'].ndim == 2 and ds['events'].shape[1] >= 4:
        ev = ds['events']
        return ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
    aliases = {
        'x': ['x', 'events/x', 'CD/events/x'],
        'y': ['y', 'events/y', 'CD/events/y'],
        't': ['t', 'events/t', 'events/ts', 'events/t_us', 'CD/events/t'],
        'p': ['p', 'events/p', 'events/polarity', 'CD/events/p'],
    }
    found = []
    for key in ['x', 'y', 't', 'p']:
        hit = next((ds[n] for n in aliases[key] if n in ds), None)
        found.append(hit)
    if all(x is not None for x in found):
        return tuple(found)
    available = '\n'.join(sorted(names))
    raise KeyError(f'Could not infer event datasets. Available H5 datasets:\n{available}')


def _read_ms_to_idx(f: h5py.File):
    for key in ['ms_to_idx', 'events/ms_to_idx']:
        if key in f:
            return f[key]
    return None


def slice_events(h5_path: Path, t_start: int, t_end: int):
    with h5py.File(h5_path, 'r') as f:
        x_ds, y_ds, t_ds, p_ds = _event_datasets(f)
        ms_to_idx = _read_ms_to_idx(f)
        if ms_to_idx is not None:
            start_ms = int(np.floor(t_start / 1000.0))
            end_ms = int(np.ceil(t_end / 1000.0))
            if end_ms < 0 or start_ms >= len(ms_to_idx):
                empty = np.array([], dtype=np.int64)
                return empty, empty, empty, empty
            start_ms = max(start_ms, 0)
            end_ms = min(end_ms, len(ms_to_idx) - 1)
            lo0 = int(ms_to_idx[start_ms])
            hi0 = int(ms_to_idx[end_ms])
            if hi0 <= lo0:
                hi0 = min(lo0 + 1, t_ds.shape[0])
        else:
            # Fallback for H5 files without ms_to_idx. This is slower but only used for small sanity subsets.
            t_all = np.asarray(t_ds, dtype=np.int64)
            lo0 = int(np.searchsorted(t_all, t_start, side='left'))
            hi0 = int(np.searchsorted(t_all, t_end, side='left'))
        tw = np.asarray(t_ds[lo0:hi0], dtype=np.int64)
        lo = lo0 + int(np.searchsorted(tw, t_start, side='left'))
        hi = lo0 + int(np.searchsorted(tw, t_end, side='left'))
        return (
            np.asarray(x_ds[lo:hi], dtype=np.int64),
            np.asarray(y_ds[lo:hi], dtype=np.int64),
            np.asarray(p_ds[lo:hi], dtype=np.int64),
            np.asarray(t_ds[lo:hi], dtype=np.int64),
        )


def old_nbin3_encode(x, y, p, t, src_hw=(720, 1280), out_hw=(480, 640), resize_mode='scale') -> np.ndarray:
    src_h, src_w = src_hw
    out_h, out_w = out_hw
    valid = (x >= 0) & (x < src_w) & (y >= 0) & (y < src_h)
    x, y, p, t = x[valid], y[valid], p[valid], t[valid]
    if resize_mode == 'center-crop-4-3':
        crop_w = min(src_w, int(round(src_h * 4.0 / 3.0)))
        x0 = max((src_w - crop_w) // 2, 0)
        keep = (x >= x0) & (x < x0 + crop_w)
        x, y, p, t = x[keep] - x0, y[keep], p[keep], t[keep]
        src_w = crop_w
    elif resize_mode != 'scale':
        raise ValueError(f'Unknown resize_mode: {resize_mode}')
    if (src_h, src_w) != (out_h, out_w) and len(x):
        x = np.floor(x.astype(np.float64) * out_w / src_w).astype(np.int64)
        y = np.floor(y.astype(np.float64) * out_h / src_h).astype(np.int64)
        x = np.clip(x, 0, out_w - 1)
        y = np.clip(y, 0, out_h - 1)
    img = np.full((3, out_h, out_w), 128, dtype=np.uint8)
    if len(t) == 0:
        return img
    if len(t) > 1:
        interval = max(int(t.max() - t.min()), 1)
        tn = (t.astype(np.float64) - float(t.min())) / float(interval)
    else:
        tn = np.zeros_like(t, dtype=np.float64)
    bins = np.minimum((tn * 3).astype(np.int64), 2)
    values = np.zeros_like(p, dtype=np.uint8)
    values[p > 0] = 255
    for b in range(3):
        cur = np.full((out_h, out_w), 128, dtype=np.uint8)
        m = bins == b
        cur[y[m], x[m]] = values[m]
        img[b] = cur
    return img


def find_sequence_files(root: Path, seq: str):
    seq_dir = root / seq
    event_h5 = _first_existing([
        seq_dir / 'events_left.h5',
        seq_dir / 'events-left.h5',
        seq_dir / 'events' / 'left' / 'events.h5',
        seq_dir / 'left_events.h5',
        seq_dir / f'{seq}-events_left.h5',
    ])
    if event_h5 is None:
        hits = sorted(seq_dir.rglob('*left*.h5')) + sorted(seq_dir.rglob('*events*.h5'))
        event_h5 = hits[0] if hits else None
    image_root = _first_existing([
        seq_dir / 'mav0' / 'cam0' / 'data',
        seq_dir / 'images' / 'left',
        seq_dir / 'images' / 'left' / 'rectified',
        seq_dir / 'left_images',
        seq_dir / 'frames' / 'left',
    ])
    if image_root is None:
        image_hits = sorted([p for p in seq_dir.rglob('*') if p.is_dir() and any(p.glob('*.png'))])
        image_root = image_hits[0] if image_hits else None
    ts_path = _first_existing([
        seq_dir / 'mav0' / 'cam0' / 'data.csv',
        seq_dir / 'images' / 'timestamps.txt',
        seq_dir / 'timestamps.txt',
        seq_dir / 'image_timestamps.txt',
        seq_dir / 'left_images' / 'image_timestamps_left.txt',
        seq_dir / 'right_images' / 'image_timestamps_right.txt',
    ])
    return event_h5, image_root, ts_path


def _event_time_range_us(event_h5: Path | None):
    if event_h5 is None or not event_h5.exists():
        return None
    with h5py.File(event_h5, 'r') as f:
        _, _, t_ds, _ = _event_datasets(f)
        return int(t_ds[0]), int(t_ds[-1])


def _coerce_timestamps_to_event_us(vals: np.ndarray, event_h5: Path | None) -> np.ndarray:
    vals = vals.astype(np.float64, copy=False)
    event_range = _event_time_range_us(event_h5)
    if event_range is None:
        return vals.astype(np.int64)
    ev0, ev1 = event_range

    def overlap_score(candidate: np.ndarray) -> float:
        return float(np.mean((candidate >= ev0) & (candidate <= ev1)))

    candidates = [
        vals,          # already microseconds, used by TUM-VIE image_timestamps_left.txt
        vals / 1000.0, # nanoseconds -> microseconds
        vals * 1e6,    # seconds -> microseconds
    ]
    best = max(candidates, key=overlap_score)
    return np.rint(best).astype(np.int64)


def load_frame_index(image_root: Path, ts_path: Path | None, event_h5: Path | None = None):
    images = sorted([*image_root.glob('*.png'), *image_root.glob('*.jpg'), *image_root.glob('*.jpeg')])
    if not images:
        raise FileNotFoundError(f'No image files found in {image_root}')
    if ts_path and ts_path.exists() and ts_path.suffix == '.csv':
        rows = []
        for line in ts_path.read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            parts = [x.strip() for x in line.split(',')]
            if not parts[0].isdigit():
                continue
            fname = parts[1] if len(parts) > 1 else f'{parts[0]}.png'
            ts = float(parts[0])
            rows.append((ts, image_root / fname))
        rows = [(t, p) for t, p in rows if p.exists()]
        if rows:
            ts_vals = _coerce_timestamps_to_event_us(np.array([r[0] for r in rows]), event_h5)
            return list(zip(ts_vals.astype(np.int64).tolist(), [r[1] for r in rows]))
    if ts_path and ts_path.exists():
        vals = np.loadtxt(ts_path, dtype=np.float64)
        vals = np.atleast_1d(vals)
        if len(vals) == len(images):
            vals = _coerce_timestamps_to_event_us(vals, event_h5)
            return list(zip(vals.astype(np.int64).tolist(), images))
    if event_h5 is not None and event_h5.exists():
        with h5py.File(event_h5, 'r') as f:
            _, _, t_ds, _ = _event_datasets(f)
            t0 = int(t_ds[0])
            t1 = int(t_ds[-1])
        if len(images) == 1:
            vals = [t0]
        else:
            vals = np.linspace(t0, t1, num=len(images), dtype=np.int64).tolist()
        return list(zip(vals, images))
    rows = []
    for p in images:
        stem = p.stem
        digits = ''.join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        ts = int(digits)
        if ts < 10_000_000_000:
            ts *= 1_000_000
        rows.append((ts, p))
    if rows:
        ts_vals = np.array([r[0] for r in rows], dtype=np.int64)
        # Plain frame-number filenames such as 00000.jpg are not timestamps.
        # TUM-VIE reports sensor timestamps in microseconds, so use them only if they
        # look like real sequence times rather than small frame indices.
        if np.nanmedian(ts_vals) > 100_000:
            return rows
    raise FileNotFoundError(f'Could not infer timestamps for images in {image_root}')


def build_trainer(cfg, checkpoint: Path, device: torch.device, base_ckpt_dir: Path, out_dir: Path):
    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    dummy_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
    model = MarigoldPipeline.from_pretrained(
        str(base_ckpt_dir / cfg.model.pretrained_path),
        **(cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}),
    )
    trainer_cls = get_trainer_cls(cfg.trainer.name)
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=dummy_loader,
        device=device,
        base_ckpt_dir=str(base_ckpt_dir),
        out_dir_ckpt=str(out_dir / '_unused_ckpt'),
        out_dir_eval=str(out_dir / '_unused_eval'),
        out_dir_vis=str(out_dir / '_unused_vis'),
        accumulation_steps=1,
        val_dataloaders=[],
        vis_dataloaders=[],
        depth_transform=depth_transform,
    )
    trainer.load_checkpoint(str(checkpoint), load_trainer_state=False, resume_lr_scheduler=False)
    return trainer, depth_transform


@torch.no_grad()
def infer_metric(trainer, depth_transform, event_chw: np.ndarray, apply_calibration: bool, return_debug: bool = False):
    trainer.model.to(trainer.device)
    trainer.model.unet.eval()
    trainer.model.vae.eval()
    trainer.model.text_encoder.eval()
    if trainer.use_calibration_head:
        trainer.calibration_head.eval()
    rgb_int = torch.from_numpy(event_chw).unsqueeze(0).int()
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(int(trainer.cfg.validation.init_seed))
    out = trainer.model(
        rgb_int,
        denoising_steps=trainer.cfg.validation.denoising_steps,
        ensemble_size=trainer.cfg.validation.ensemble_size,
        processing_res=trainer.cfg.validation.processing_res,
        match_input_res=trainer.cfg.validation.match_input_res,
        generator=generator,
        batch_size=1,
        color_map='magma',
        show_progress_bar=False,
        resample_method=trainer.cfg.validation.resample_method,
    )
    norm_min = float(trainer.cfg.depth_normalization.norm_min)
    norm_max = float(trainer.cfg.depth_normalization.norm_max)
    pred_norm = out.depth_np * (norm_max - norm_min) + norm_min
    depth = depth_transform.denormalize(torch.from_numpy(pred_norm).float()).cpu().numpy()
    depth = np.clip(depth, 1e-6, None)
    debug = {
        'raw_metric': depth.astype(np.float32),
        'calibration_scale': None,
        'calibration_shift': None,
    }
    if apply_calibration and trainer.use_calibration_head:
        depth_ts = (
            torch.from_numpy(depth)
            .to(trainer.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scale, shift = trainer.calibration_head(depth_ts)
        depth = (scale[:, None, None, None] * depth_ts + shift[:, None, None, None])
        debug['calibration_scale'] = float(scale.detach().cpu().item())
        debug['calibration_shift'] = float(shift.detach().cpu().item())
        depth = depth.squeeze(0).squeeze(0).detach().cpu().numpy()
    depth = np.clip(depth, 1e-6, None)
    debug['final_metric'] = depth.astype(np.float32)
    if return_debug:
        return depth, debug
    return depth


def save_sanity_panel(frame_rgb: np.ndarray, event_chw: np.ndarray, out_path: Path):
    event_rgb = np.moveaxis(event_chw, 0, -1)
    frame_resized = cv2.resize(frame_rgb, (event_rgb.shape[1], event_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([frame_resized, event_rgb], axis=1)
    save_rgb_svg(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser(description='TUM-VIE zero-shot converter and qualitative inference for DSEC-trained Marigold.')
    ap.add_argument('--tumvie-root', type=Path, default=Path('/path/to/data/event3d/TUM-VIE/subset'))
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out-root', type=Path, default=Path('/path/to/user/code/paper_assets/tumvie_zero_shot'))
    ap.add_argument('--stage1-checkpoint', type=Path, default=DEFAULT_STAGE1)
    ap.add_argument('--stage2-checkpoint', type=Path, default=DEFAULT_STAGE2)
    ap.add_argument('--calibrated-only', action='store_true', help='Run only the stage-II calibrated checkpoint.')
    ap.add_argument('--debug-calibration', action='store_true', help='Also save raw pre-calibration depth and calibration scale/shift.')
    ap.add_argument('--base-ckpt-dir', type=Path, default=Path(os.environ.get('BASE_CKPT_DIR', '/path/to/models')))
    ap.add_argument('--frames-per-seq', type=int, default=6)
    ap.add_argument('--start-frac', type=float, default=0.1)
    ap.add_argument('--end-frac', type=float, default=0.9)
    ap.add_argument('--window-mode', choices=['frame-interval', 'dsec-two-frame', 'fixed-us'], default='frame-interval')
    ap.add_argument('--window-us', type=int, default=100_000)
    ap.add_argument('--src-height', type=int, default=720)
    ap.add_argument('--src-width', type=int, default=1280)
    ap.add_argument('--out-height', type=int, default=None)
    ap.add_argument('--out-width', type=int, default=None)
    ap.add_argument('--resize-mode', choices=['scale', 'center-crop-4-3'], default='scale')
    ap.add_argument('--sanity-only', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.out_height is None:
        args.out_height = args.src_height
    if args.out_width is None:
        args.out_width = args.src_width
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    trainers = []
    if not args.sanity_only:
        stage_specs = [('stage2_calibrated', args.stage2_checkpoint)] if args.calibrated_only else [('stage1', args.stage1_checkpoint), ('stage2', args.stage2_checkpoint)]
        for name, ckpt in stage_specs:
            cfg = OmegaConf.load(ckpt.parent.parent / 'config.yaml')
            tr, dt = build_trainer(cfg, ckpt, device, args.base_ckpt_dir, args.out_root)
            trainers.append((name, tr, dt, name.startswith('stage2')))

    for seq in args.seqs:
        seq_out = args.out_root / seq
        nbin_out = seq_out / 'nbin3'
        sanity_out = seq_out / 'sanity'
        pred_out = seq_out / 'predictions'
        nbin_out.mkdir(parents=True, exist_ok=True)
        sanity_out.mkdir(parents=True, exist_ok=True)
        pred_out.mkdir(parents=True, exist_ok=True)

        event_h5, image_root, ts_path = find_sequence_files(args.tumvie_root, seq)
        if event_h5 is None or image_root is None:
            print(f'[{seq}] missing event_h5={event_h5} image_root={image_root}; expected downloaded subset under {args.tumvie_root / seq}')
            continue
        frames = load_frame_index(image_root, ts_path, event_h5)
        lo = max(2, int(len(frames) * args.start_frac))
        hi = max(lo + 1, int(len(frames) * args.end_frac))
        ids = np.linspace(lo, hi - 1, num=min(args.frames_per_seq, hi - lo), dtype=int)
        print(f'[{seq}] event_h5={event_h5} images={len(frames)} selected={ids.tolist()}')

        for frame_idx in ids:
            t_end, img_path = frames[int(frame_idx)]
            if args.window_mode == 'frame-interval' and frame_idx > 0:
                # TUM-VIE grayscale frames are 20 Hz, so adjacent image timestamps define
                # the natural per-frame event packet (~50 ms).
                t_start = int(frames[int(frame_idx) - 1][0])
            elif args.window_mode == 'dsec-two-frame' and frame_idx >= 2:
                # Kept only for reproducing the DSEC old-nbin convention, where one
                # nbin sample corresponds to every other RGB frame.
                t_start = int(frames[int(frame_idx) - 2][0])
            elif args.window_mode == 'dsec-two-frame' and frame_idx > 0:
                t_start = int(frames[int(frame_idx) - 1][0])
            else:
                t_start = int(t_end) - int(args.window_us)
            x, y, p, t = slice_events(event_h5, int(t_start), int(t_end))
            event = old_nbin3_encode(
                x, y, p, t,
                src_hw=(args.src_height, args.src_width),
                out_hw=(args.out_height, args.out_width),
                resize_mode=args.resize_mode,
            )
            stem = f'{frame_idx:06d}'
            iio.imwrite(nbin_out / f'{stem}.tif', event)
            frame_rgb = np.array(Image.open(img_path).convert('RGB'))
            save_sanity_panel(frame_rgb, event, sanity_out / f'{stem}_frame_event.svg')
            frame_dir = pred_out / stem
            frame_dir.mkdir(exist_ok=True)

            panels = {
                'reference.svg': cv2.resize(frame_rgb, (args.out_width, args.out_height), interpolation=cv2.INTER_AREA),
                'event.svg': np.moveaxis(event, 0, -1),
            }
            if not args.sanity_only:
                debug_lines = []
                for name, tr, dt, use_cal in trainers:
                    pred_result = infer_metric(
                        tr, dt, event, apply_calibration=use_cal, return_debug=args.debug_calibration
                    )
                    if args.debug_calibration:
                        pred, debug = pred_result
                        np.save(frame_dir / f'prediction_{name}_raw_before_calibration.npy', debug['raw_metric'])
                        panels[f'prediction_{name}_raw_before_calibration.svg'] = inverse_log_magma_numpy(debug['raw_metric'])
                        debug_lines.extend([
                            f'{name}_calibration_scale: {debug["calibration_scale"]}',
                            f'{name}_calibration_shift: {debug["calibration_shift"]}',
                            f'{name}_raw_min_median_max: {float(np.nanmin(debug["raw_metric"])):.6g} {float(np.nanmedian(debug["raw_metric"])):.6g} {float(np.nanmax(debug["raw_metric"])):.6g}',
                            f'{name}_final_min_median_max: {float(np.nanmin(debug["final_metric"])):.6g} {float(np.nanmedian(debug["final_metric"])):.6g} {float(np.nanmax(debug["final_metric"])):.6g}',
                        ])
                    else:
                        pred = pred_result
                    np.save(frame_dir / f'prediction_{name}.npy', pred.astype(np.float32))
                    panels[f'prediction_{name}.svg'] = inverse_log_magma_numpy(pred)
            for fname, panel in panels.items():
                save_rgb_svg(str(frame_dir / fname), panel)
            meta = (
                f'seq: {seq}\nframe_index: {frame_idx}\nimage_path: {img_path}\n'
                f'event_h5: {event_h5}\nevent_window_us: {t_start} {t_end}\n'
                f'event_count: {len(x)}\nwindow_mode: {args.window_mode}\n'
                f'src_hw: {args.src_height},{args.src_width}\nout_hw: {args.out_height},{args.out_width}\n'
                f'resize_mode: {args.resize_mode}\n'
            )
            if not args.sanity_only and args.debug_calibration and 'debug_lines' in locals():
                meta += '\n'.join(debug_lines) + ('\n' if debug_lines else '')
            (frame_dir / 'metadata.txt').write_text(meta)

    print(f'Done. Output root: {args.out_root}')


if __name__ == '__main__':
    main()
