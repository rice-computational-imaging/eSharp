#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTHANYEVENT_ROOT = Path("/path/to/user/code/depthanyevent")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEPTHANYEVENT_ROOT))

from marigold.marigold_pipeline import MarigoldPipeline
from models.dav2 import get_depth_anything_v2
from src.dataset import DatasetMode, get_dataset
from src.trainer import get_trainer_cls
from src.util.depth_transform import get_depth_normalizer
from src.util.logging_util import inverse_log_magma_numpy, save_rgb_svg


def center_crop_hw(arr: np.ndarray, crop_hw: tuple[int, int]) -> np.ndarray:
    crop_h, crop_w = crop_hw
    h, w = arr.shape[:2]
    y0 = max((h - crop_h) // 2, 0)
    x0 = max((w - crop_w) // 2, 0)
    return arr[y0 : y0 + min(crop_h, h), x0 : x0 + min(crop_w, w)]


def resize_rgb_to_event_canvas(rgb: np.ndarray, event_hw: tuple[int, int]) -> np.ndarray:
    h, w = event_hw
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


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


def load_left_rgb(base_data_dir: Path, seq: str, frame_id: int, crop_hw: tuple[int, int] | None) -> np.ndarray:
    rgb_idx = frame_id * 2
    rgb_path = (
        base_data_dir
        / "event3d/DSEC/train/rgb_frames"
        / seq
        / "images/left/rectified"
        / f"{rgb_idx:06d}.png"
    )
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read RGB frame: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_event_canvas = resize_rgb_to_event_canvas(rgb, (480, 640))
    if crop_hw is None:
        return rgb_event_canvas
    return center_crop_hw(rgb_event_canvas, crop_hw)


@torch.no_grad()
def infer_depthanything_raw(
    rgb_u8: np.ndarray,
    device: torch.device,
    weight_path: Path,
    config_path: Path,
) -> np.ndarray:
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
    rgb_t = (
        torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    pred = model.infer_image(rgb_t)
    return pred.squeeze().detach().cpu().numpy().astype(np.float32)


def build_one_frame_loader(
    cfg,
    base_data_dir: Path,
    split_file: Path,
    depth_transform,
):
    ds_cfg = OmegaConf.create(
        {
            "name": "dsec_event_disparity",
            "disp_name": "dsec_pipeline_one_frame",
            "dir": ".",
            "filenames": str(split_file),
        }
    )
    event_input_mode = OmegaConf.select(cfg, "dataset.train.event_input_mode", default="tenacode")
    dataset = get_dataset(
        ds_cfg,
        base_data_dir=str(base_data_dir),
        mode=DatasetMode.EVAL,
        depth_transform=depth_transform,
        augmentation_args=cfg.augmentation,
        event_input_mode=event_input_mode,
    )
    return DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)


def build_trainer_for_checkpoint(
    cfg,
    checkpoint: Path,
    loader,
    device: torch.device,
    base_ckpt_dir: Path,
    depth_transform,
    out_dir: Path,
):
    model = MarigoldPipeline.from_pretrained(
        str(base_ckpt_dir / cfg.model.pretrained_path),
        **(cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}),
    )
    trainer_cls = get_trainer_cls(cfg.trainer.name)
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=loader,
        device=device,
        base_ckpt_dir=str(base_ckpt_dir),
        out_dir_ckpt=str(out_dir / "_unused_ckpt"),
        out_dir_eval=str(out_dir / "_unused_eval"),
        out_dir_vis=str(out_dir / "_unused_vis"),
        accumulation_steps=1,
        val_dataloaders=[loader],
        vis_dataloaders=[loader],
        depth_transform=depth_transform,
    )
    trainer.load_checkpoint(str(checkpoint), load_trainer_state=False, resume_lr_scheduler=False)
    return trainer


@torch.no_grad()
def infer_marigold_metric(trainer, batch: dict, dataset, apply_calibration: bool) -> np.ndarray:
    trainer.model.to(trainer.device)
    trainer.model.unet.eval()
    trainer.model.vae.eval()
    trainer.model.text_encoder.eval()
    if trainer.use_calibration_head:
        trainer.calibration_head.eval()

    rgb_int = batch["rgb_int"]
    depth_raw = batch["depth_raw_linear"].squeeze().numpy()
    valid_mask = batch["valid_mask_raw"].squeeze().numpy()

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
    pred = trainer._pred_to_metric_for_eval(pipe_out.depth_np, depth_raw, valid_mask)
    if apply_calibration and trainer.use_calibration_head and trainer.calibration_apply_in_validation:
        pred = trainer._apply_calibration_head_metric(pred)

    pred = np.clip(pred, dataset.min_depth, dataset.max_depth)
    return np.clip(pred, 1e-6, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final DSEC pipeline SVG panels.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/path/to/repository/Marigold/output/train_marigold_dsec_old_nbin_calib_joint_/checkpoint/latest"),
    )
    parser.add_argument(
        "--no-cal-checkpoint",
        type=Path,
        default=Path("/path/to/repository/Marigold/output/train_marigold_dsec_old_nbin_/checkpoint/latest"),
        help="Pre-calibration Marigold checkpoint used for prediction_without_calibration_head.svg.",
    )
    parser.add_argument("--seq", default="interlaken_00_f")
    parser.add_argument("--frame-id", type=int, default=20)
    parser.add_argument("--base-data-dir", type=Path, default=Path(os.environ.get("BASE_DATA_DIR", "/path/to/data")))
    parser.add_argument(
        "--base-ckpt-dir",
        type=Path,
        default=Path(os.environ.get("BASE_CKPT_DIR", "/path/to/models")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/user/code/paper_assets/dsec_pipeline_interlaken_00_f_0000000020"),
    )
    parser.add_argument(
        "--dav2-weight",
        type=Path,
        default=Path("/path/to/user/code/depthanyevent/weights/finetuned_dsec/finetuned_dsec.pth"),
    )
    parser.add_argument(
        "--dav2-config",
        type=Path,
        default=Path("/path/to/user/code/depthanyevent/weights/finetuned_dsec/config.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--svg-dpi", type=int, default=300)
    args = parser.parse_args()

    calib_run_dir = args.checkpoint.parent.parent
    no_cal_run_dir = args.no_cal_checkpoint.parent.parent
    cfg = OmegaConf.load(calib_run_dir / "config.yaml")
    cfg_no_cal = OmegaConf.load(no_cal_run_dir / "config.yaml")
    crop_hw_cfg = OmegaConf.select(cfg, "augmentation.center_crop_hw", default=None)
    crop_hw = tuple(crop_hw_cfg) if crop_hw_cfg is not None else None
    depth_vis_min_m = float(OmegaConf.select(cfg, "eval.vis_magma_min_m", default=2.0))
    depth_vis_max_m = float(OmegaConf.select(cfg, "eval.vis_magma_max_m", default=100.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    disparity_frame = args.frame_id * 2
    split_line = (
        f"event3d/DSEC/train/nbin_3_encoding/{args.seq}/{args.frame_id:010d}.tif "
        f"dsec_generative_label_event_disparity_affine/{args.seq}/frame{disparity_frame:06d}_pred_depth.npy\n"
    )

    split_path = args.out_dir / "_one_frame_split.txt"
    split_path.write_text(split_line)

    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    loader = build_one_frame_loader(cfg, args.base_data_dir, split_path, depth_transform)
    batch = next(iter(loader))
    dataset = loader.dataset

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    trainer_cal = build_trainer_for_checkpoint(
        cfg, args.checkpoint, loader, device, args.base_ckpt_dir, depth_transform, args.out_dir
    )
    trainer_no_cal = build_trainer_for_checkpoint(
        cfg_no_cal,
        args.no_cal_checkpoint,
        loader,
        device,
        args.base_ckpt_dir,
        depth_transform,
        args.out_dir,
    )

    depth_gt = batch["depth_raw_linear"].squeeze().numpy()
    pseudo_metric = batch["pseudo_depth_linear"].squeeze().numpy()
    event_rgb = np.moveaxis(batch["rgb_int"].detach().cpu().numpy().squeeze(), 0, -1)
    rgb_u8 = load_left_rgb(args.base_data_dir, args.seq, args.frame_id, crop_hw)
    pred_no_cal = infer_marigold_metric(
        trainer_no_cal, batch, dataset, apply_calibration=False
    )
    pred_with_cal = infer_marigold_metric(
        trainer_cal, batch, dataset, apply_calibration=True
    )
    dav2_raw = infer_depthanything_raw(rgb_u8, device, args.dav2_weight, args.dav2_config)

    panels = {
        "event.svg": event_rgb,
        "rgb.svg": rgb_u8,
        "gt_depth.svg": inverse_log_magma_numpy(depth_gt, depth_vis_min_m, depth_vis_max_m),
        "raw_pseudolabel_depthanythingv2.svg": relative_magma_u8(dav2_raw),
        "metric_aligned_pseudolabel.svg": inverse_log_magma_numpy(
            pseudo_metric, depth_vis_min_m, depth_vis_max_m
        ),
        "prediction_without_calibration_head.svg": inverse_log_magma_numpy(
            pred_no_cal, depth_vis_min_m, depth_vis_max_m
        ),
        "prediction_with_calibration_head.svg": inverse_log_magma_numpy(
            pred_with_cal, depth_vis_min_m, depth_vis_max_m
        ),
    }

    for name, panel in panels.items():
        save_rgb_svg(str(args.out_dir / name), panel, dpi=args.svg_dpi)

    print(f"Saved {len(panels)} SVG panels to {args.out_dir}")
    print(f"Calibration checkpoint: {args.checkpoint}")
    print(f"No-calibration checkpoint: {args.no_cal_checkpoint}")
    print(f"Frame: {args.seq} event={args.frame_id:010d} disparity/rgb={disparity_frame:06d}")
    print(f"RGB crop: {crop_hw if crop_hw is not None else 'none'}")


if __name__ == "__main__":
    main()
