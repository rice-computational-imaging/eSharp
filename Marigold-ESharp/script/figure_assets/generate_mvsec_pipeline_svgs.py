#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from marigold.marigold_pipeline import MarigoldPipeline
from src.dataset import DatasetMode, get_dataset
from src.trainer import get_trainer_cls
from src.util.depth_transform import get_depth_normalizer
from src.util.logging_util import get_visualization_panels, save_rgb_svg


def resize_rgb_to_panel(rgb: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def load_mvsec_rgb(base_data_dir: Path, seq: str, frame_id: int, rgb_folder: str) -> np.ndarray:
    rgb_path = (
        base_data_dir
        / "event3d/mvsec/train"
        / seq
        / "rgb"
        / rgb_folder
        / f"frame_{frame_id:010d}.png"
    )
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read MVSEC RGB frame: {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_one_frame_loader(cfg, base_data_dir: Path, split_file: Path, depth_transform):
    ds_cfg = OmegaConf.create(
        {
            "name": "mvsec_gt",
            "disp_name": "mvsec_pipeline_one_frame",
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
def infer_marigold_metric(trainer, batch: dict, dataset) -> np.ndarray:
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
    if trainer.use_calibration_head and trainer.calibration_apply_in_validation:
        pred = trainer._apply_calibration_head_metric(pred)
    pred = np.clip(pred, dataset.min_depth, dataset.max_depth)
    return np.clip(pred, 1e-6, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final MVSEC pipeline SVG panels.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/path/to/user/marigold/26_07_07-02_58_33-train_marigold_mvsec_pesudo_nbin3_mvsec_nocalib_quantile/checkpoint/latest"
        ),
    )
    parser.add_argument("--seq", default="mvsec_outdoor_night1")
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--rgb-folder", default="davis_left")
    parser.add_argument("--base-data-dir", type=Path, default=Path(os.environ.get("BASE_DATA_DIR", "/path/to/data")))
    parser.add_argument(
        "--base-ckpt-dir",
        type=Path,
        default=Path(os.environ.get("BASE_CKPT_DIR", "/path/to/models")),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--svg-dpi", type=int, default=300)
    args = parser.parse_args()

    run_dir = args.checkpoint.parent.parent
    cfg = OmegaConf.load(run_dir / "config.yaml")
    depth_vis_min_m = float(OmegaConf.select(cfg, "eval.vis_magma_min_m", default=2.0))
    depth_vis_max_m = float(OmegaConf.select(cfg, "eval.vis_magma_max_m", default=100.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_line = (
        f"event3d/MVSEC_regen/nbin3/{args.seq}/{args.frame_id:010d}.tif "
        f"event3d/MVSEC_regen/pesudo_depth_quantile_shift/{args.seq}/frame_{args.frame_id:010d}_pred_depth.npy\n"
    )
    split_path = args.out_dir / "_one_frame_split.txt"
    split_path.write_text(split_line)

    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    loader = build_one_frame_loader(cfg, args.base_data_dir, split_path, depth_transform)
    batch = next(iter(loader))
    dataset = loader.dataset

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    trainer = build_trainer_for_checkpoint(
        cfg, args.checkpoint, loader, device, args.base_ckpt_dir, depth_transform, args.out_dir
    )

    depth_gt = batch["depth_raw_linear"].squeeze().numpy()
    valid_mask = batch["valid_mask_raw"].squeeze().numpy()
    pseudo_metric = batch["pseudo_depth_linear"].squeeze().numpy()
    event_rgb = np.moveaxis(batch["rgb_int"].detach().cpu().numpy().squeeze(), 0, -1)
    pred = infer_marigold_metric(trainer, batch, dataset)

    rgb_u8 = load_mvsec_rgb(args.base_data_dir, args.seq, args.frame_id, args.rgb_folder)
    rgb_u8 = resize_rgb_to_panel(rgb_u8, depth_gt.shape[:2])

    panels = get_visualization_panels(
        event_rgb,
        depth_gt,
        pred,
        pseudo_depth=pseudo_metric,
        valid_mask=valid_mask,
        depth_vis_min_m=depth_vis_min_m,
        depth_vis_max_m=depth_vis_max_m,
    )
    outputs = {
        "event.svg": panels["event"],
        "rgb.svg": rgb_u8,
        "gt_depth.svg": panels["ground_truth"],
        "pseudo_depth.svg": panels["pseudo_label"],
        "prediction.svg": panels["prediction"],
    }
    for name, panel in outputs.items():
        save_rgb_svg(str(args.out_dir / name), panel, dpi=args.svg_dpi)

    print(f"Saved {len(outputs)} SVG panels to {args.out_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Frame: {args.seq} {args.frame_id:010d}")
    print(f"RGB folder: {args.rgb_folder}")


if __name__ == "__main__":
    main()
