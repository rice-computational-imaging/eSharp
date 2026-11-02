#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401 - registers compressed DSEC HDF5 filters
import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTHANYEVENT_ROOT = Path("/path/to/user/code/depthanyevent")
EVENT_WASSDIFF_ROOT = Path("/path/to/event-wassdiff")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEPTHANYEVENT_ROOT))
sys.path.append(str(EVENT_WASSDIFF_ROOT))

from marigold.marigold_pipeline import MarigoldPipeline
from models.dav2 import get_depth_anything_v2
from align_rgb_to_event_camera import _load_calib, _parse_vec3, warp_rgb_via_plane_homography
from src.trainer import get_trainer_cls
from src.util.depth_transform import get_depth_normalizer
from src.util.logging_util import inverse_log_magma_numpy, save_rgb_svg


def center_crop_hw(arr: np.ndarray, crop_hw: tuple[int, int] | None) -> np.ndarray:
    if crop_hw is None:
        return arr
    crop_h, crop_w = crop_hw
    h, w = arr.shape[:2]
    y0 = max((h - crop_h) // 2, 0)
    x0 = max((w - crop_w) // 2, 0)
    return arr[y0 : y0 + min(crop_h, h), x0 : x0 + min(crop_w, w)]


def load_rgb(base_data_dir: Path, seq: str, frame_id: int, crop_hw: tuple[int, int] | None) -> np.ndarray:
    rgb_path = (
        base_data_dir
        / "event3d/DSEC/test/rgb_frames"
        / seq
        / "images/left/rectified"
        / f"{frame_id:06d}.png"
    )
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read RGB frame: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return center_crop_hw(rgb, crop_hw)


def project_rgb_to_event_plane(
    rgb: np.ndarray,
    base_data_dir: Path,
    seq: str,
    event_hw: tuple[int, int] = (480, 640),
    plane_normal: str = "0,0,1",
    plane_d: float = -10.0,
) -> np.ndarray:
    calib_path = base_data_dir / "event3d/DSEC/test/calib" / seq / "calibration/cam_to_cam.yaml"
    calib = _load_calib(str(calib_path))
    h_e, w_e = event_hw
    return warp_rgb_via_plane_homography(
        rgb,
        calib,
        event_size=(w_e, h_e),
        rgb_is_rectified=True,
        plane_normal=_parse_vec3(plane_normal),
        plane_d=plane_d,
    )


def read_image_timestamps(base_data_dir: Path, seq: str) -> np.ndarray:
    ts_path = base_data_dir / "event3d/DSEC/test/rgb_frames" / seq / "images/timestamps.txt"
    return np.loadtxt(ts_path, dtype=np.int64)


def slice_events(h5_path: Path, t_start: int, t_end: int):
    with h5py.File(h5_path, "r") as f:
        t_offset = int(np.asarray(f["t_offset"]).item()) if "t_offset" in f else 0
        t_start_local = int(t_start - t_offset)
        t_end_local = int(t_end - t_offset)
        start_ms = max(int(np.floor(t_start_local / 1000.0)), 0)
        end_ms = int(np.ceil(t_end_local / 1000.0))
        ms_to_idx = f["ms_to_idx"]
        if start_ms >= len(ms_to_idx):
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        end_ms = min(end_ms, len(ms_to_idx) - 1)
        lo0 = int(ms_to_idx[start_ms])
        hi0 = int(ms_to_idx[end_ms])
        if hi0 <= lo0:
            hi0 = min(lo0 + 1, f["events/t"].shape[0])
        t_window = np.asarray(f["events/t"][lo0:hi0], dtype=np.int64)
        lo = lo0 + int(np.searchsorted(t_window, t_start_local, side="left"))
        hi = lo0 + int(np.searchsorted(t_window, t_end_local, side="left"))
        events = f["events"]
        return (
            np.asarray(events["x"][lo:hi], dtype=np.int64),
            np.asarray(events["y"][lo:hi], dtype=np.int64),
            np.asarray(events["p"][lo:hi], dtype=np.int64),
            np.asarray(events["t"][lo:hi], dtype=np.int64),
        )


def render_event_rgb(
    base_data_dir: Path,
    seq: str,
    frame_id: int,
    crop_hw: tuple[int, int] | None,
    window_us: int | None,
) -> tuple[np.ndarray, tuple[int, int], int]:
    del crop_hw
    timestamps = read_image_timestamps(base_data_dir, seq)
    # Old DSEC nbin3 files are indexed at half the RGB/pseudo frame id:
    # nbin frame N corresponds to RGB frame 2*N. Use the previous old-nbin
    # sample boundary, i.e. a two-RGB-frame event window by default.
    t_end = int(timestamps[frame_id])
    if window_us is None and frame_id >= 2:
        t_start = int(timestamps[frame_id - 2])
    elif window_us is None and frame_id > 0:
        t_start = int(timestamps[frame_id - 1])
    else:
        t_start = t_end - int(window_us if window_us is not None else 100_000)

    h5_path = base_data_dir / "event3d/DSEC/test/events" / seq / "events/left/events.h5"
    x, y, p, t = slice_events(h5_path, t_start, t_end)

    # Old nbin3 uint8 CHW event image. Each channel is one temporal bin:
    # 128 = no event, 255 = positive event, 0 = negative event. Marigold visualizes
    # these three bins directly as RGB channels through the normal image path.
    img = np.full((3, 480, 640), 128, dtype=np.uint8)
    valid = (x >= 0) & (x < 640) & (y >= 0) & (y < 480)
    x, y, p, t = x[valid], y[valid], p[valid], t[valid]
    if len(x) == 0:
        return img, (t_start, t_end), 0


    # Match Event-WassDiff nbin_encoding(): normalize by the selected event slice itself,
    # not by the outer timestamp window boundaries.
    if len(t) > 1:
        interval = max(int(t.max() - t.min()), 1)
        normalized_time = (t.astype(np.float64) - float(t.min())) / float(interval)
    else:
        normalized_time = np.zeros_like(t, dtype=np.float64)
    bins = np.minimum((normalized_time * 3).astype(np.int64), 2)

    values = p.copy().astype(np.int64)
    values[values == 1] = 255
    values[values == -1] = 0
    values[values == 0] = 0
    values = values.astype(np.uint8)
    for b in range(3):
        cur = np.full((480, 640), 128, dtype=np.uint8)
        m = bins == b
        cur[y[m], x[m]] = values[m]
        img[b] = cur

    return img, (t_start, t_end), int(len(x))


def relative_magma_u8(relative_depth: np.ndarray, q_low: float = 0.02, q_high: float = 0.98) -> np.ndarray:
    depth = np.asarray(relative_depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.quantile(depth[finite], [q_low, q_high])
    denom = max(float(hi - lo), 1e-6)
    unit = np.clip((depth - float(lo)) / denom, 0.0, 1.0)
    import matplotlib.pyplot as plt

    rgb = plt.cm.magma(unit)[:, :, :3]
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    rgb_u8[~finite] = 0
    return rgb_u8


@torch.no_grad()
def infer_depthanything_raw(rgb_u8: np.ndarray, device: torch.device, weight_path: Path, config_path: Path) -> np.ndarray:
    with config_path.open("r") as f:
        cfg = json.load(f)["model"]
    model = get_depth_anything_v2(
        checkpoint_path=str(weight_path),
        encoder=cfg["encoder"],
        activation=cfg["activation"],
        scale_factor=cfg["scale_factor"],
        inv_prediction=cfg["inv_prediction"],
        input_size_width=cfg["input_size_width"],
        input_size_height=cfg["input_size_height"],
        freeze_encoder=cfg.get("freeze_encoder", False),
        input_channels=cfg.get("input_channels", 3),
    ).to(device)
    model.eval()
    rgb_t = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    pred = model.infer_image(rgb_t)
    return pred.squeeze().detach().cpu().numpy().astype(np.float32)


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
        out_dir_ckpt=str(out_dir / "_unused_ckpt"),
        out_dir_eval=str(out_dir / "_unused_eval"),
        out_dir_vis=str(out_dir / "_unused_vis"),
        accumulation_steps=1,
        val_dataloaders=[],
        vis_dataloaders=[],
        depth_transform=depth_transform,
    )
    trainer.load_checkpoint(str(checkpoint), load_trainer_state=False, resume_lr_scheduler=False)
    return trainer, depth_transform


@torch.no_grad()
def infer_marigold_metric(trainer, depth_transform, event_rgb: np.ndarray, apply_calibration: bool) -> np.ndarray:
    trainer.model.to(trainer.device)
    trainer.model.unet.eval()
    trainer.model.vae.eval()
    trainer.model.text_encoder.eval()
    if trainer.use_calibration_head:
        trainer.calibration_head.eval()

    if event_rgb.ndim == 3 and event_rgb.shape[0] == 3:
        rgb_int = torch.from_numpy(event_rgb).unsqueeze(0).int()
    else:
        rgb_int = torch.from_numpy(event_rgb).permute(2, 0, 1).unsqueeze(0).int()
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(int(trainer.cfg.validation.init_seed))

    pipe_out = trainer.model(
        rgb_int,
        denoising_steps=trainer.cfg.validation.denoising_steps,
        ensemble_size=trainer.cfg.validation.ensemble_size,
        processing_res=trainer.cfg.validation.processing_res,
        match_input_res=trainer.cfg.validation.match_input_res,
        generator=generator,
        batch_size=1,
        color_map="magma",
        show_progress_bar=False,
        resample_method=trainer.cfg.validation.resample_method,
    )
    norm_min = float(trainer.cfg.depth_normalization.norm_min)
    norm_max = float(trainer.cfg.depth_normalization.norm_max)
    depth_pred_norm = pipe_out.depth_np * (norm_max - norm_min) + norm_min
    depth_metric = depth_transform.denormalize(torch.from_numpy(depth_pred_norm).float()).cpu().numpy()
    depth_metric = np.clip(depth_metric, 1e-6, None)
    if apply_calibration and trainer.use_calibration_head:
        depth_metric = trainer._apply_calibration_head_metric(depth_metric)
    return np.clip(depth_metric, 1e-6, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DSEC official-test SVG panels without GT.")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--frame-id", type=int, required=True, help="RGB frame id, e.g. 950.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/path/to/repository/Marigold/output/train_marigold_dsec_old_nbin_calib_joint_/checkpoint/latest"))
    parser.add_argument("--base-data-dir", type=Path, default=Path(os.environ.get("BASE_DATA_DIR", "/path/to/data")))
    parser.add_argument("--base-ckpt-dir", type=Path, default=Path(os.environ.get("BASE_CKPT_DIR", "/path/to/models")))
    parser.add_argument("--window-us", type=int, default=None)
    parser.add_argument("--plane-normal", default="0,0,1")
    parser.add_argument("--plane-d", type=float, default=-10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--svg-dpi", type=int, default=300)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.checkpoint.parent.parent / "config.yaml")
    crop_hw = None
    depth_vis_min_m = float(OmegaConf.select(cfg, "eval.vis_magma_min_m", default=2.0))
    depth_vis_max_m = float(OmegaConf.select(cfg, "eval.vis_magma_max_m", default=100.0))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rgb_raw = load_rgb(args.base_data_dir, args.seq, args.frame_id, crop_hw)
    rgb = project_rgb_to_event_plane(
        rgb_raw,
        args.base_data_dir,
        args.seq,
        plane_normal=args.plane_normal,
        plane_d=args.plane_d,
    )
    event_rgb, event_window, event_count = render_event_rgb(
        args.base_data_dir, args.seq, args.frame_id, crop_hw, args.window_us
    )
    iio.imwrite(args.out_dir / f"old_nbin3_{args.frame_id // 2:010d}.tif", event_rgb)

    trainer_cal, dt_cal = build_trainer(cfg, args.checkpoint, device, args.base_ckpt_dir, args.out_dir)
    pred_with_cal = infer_marigold_metric(trainer_cal, dt_cal, event_rgb, apply_calibration=True)
    event_panel = np.moveaxis(event_rgb, 0, -1) if event_rgb.ndim == 3 and event_rgb.shape[0] == 3 else event_rgb
    panels = {
        "event.svg": event_panel,
        "rgb.svg": rgb,
        "prediction.svg": inverse_log_magma_numpy(
            pred_with_cal, depth_vis_min_m, depth_vis_max_m
        ),
    }
    for name, panel in panels.items():
        save_rgb_svg(str(args.out_dir / name), panel, dpi=args.svg_dpi)

    meta = (
        f"seq: {args.seq}\n"
        f"rgb_frame_id: {args.frame_id:06d}\n"
        f"old_nbin_frame_id: {args.frame_id // 2:010d}\n"
        f"event_window_us: {event_window[0]} {event_window[1]}\n"
        f"event_count: {event_count}\n"
        f"rgb_crop: {crop_hw if crop_hw is not None else 'none'}\n"
        f"rgb_projection: Event-WassDiff warp_rgb_via_plane_homography\n"
        f"plane_normal: {args.plane_normal}\n"
        f"plane_d: {args.plane_d}\n"
        f"checkpoint: {args.checkpoint}\n"
    )
    (args.out_dir / "metadata.txt").write_text(meta)
    print(f"Saved {len(panels)} SVG panels to {args.out_dir}")
    print(meta)


if __name__ == "__main__":
    main()
