#!/usr/bin/env python3
"""Export one DSEC old-nbin3 frame with a Marigold checkpoint and DA2 labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from generate_dsec_pipeline_svgs import (
    build_one_frame_loader,
    build_trainer_for_checkpoint,
    infer_marigold_metric,
    load_left_rgb,
)
from src.util.depth_transform import get_depth_normalizer
from src.util.logging_util import (
    get_visualization_panels,
    inverse_log_magma_numpy,
    save_rgb_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DA2-label DSEC SVG panels for one Marigold checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/path/to/data/marigold/26_07_10-11_33_00-"
            "train_marigold_dsec_old_nbin_calib_joint_calib_crop/checkpoint/latest"
        ),
    )
    parser.add_argument("--seq", default="zurich_city_07_a")
    parser.add_argument("--event-frame-id", type=int, default=125)
    parser.add_argument(
        "--pseudo-root",
        type=Path,
        default=Path("/path/to/user/git/Diffused-Remap/outputs/dsec_da2_lidar_pseudo_labels"),
    )
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path(os.environ.get("BASE_DATA_DIR", "/path/to/data")),
    )
    parser.add_argument(
        "--base-ckpt-dir",
        type=Path,
        default=Path(
            os.environ.get("BASE_CKPT_DIR", "/path/to/models")
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/path/to/user/code/paper_assets/"
            "dsec_da2_lidar_zurich_city_07_a_event_0000000125"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--svg-dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    pseudo_frame_id = args.event_frame_id * 2
    pseudo_path = args.pseudo_root / args.seq / f"frame{pseudo_frame_id}_pred_depth.npy"
    if not pseudo_path.is_file():
        raise FileNotFoundError(f"Missing DA2 pseudo label: {pseudo_path}")

    cfg = OmegaConf.load(checkpoint.parent.parent / "config.yaml")
    crop_hw_cfg = OmegaConf.select(cfg, "augmentation.center_crop_hw", default=None)
    crop_hw = tuple(crop_hw_cfg) if crop_hw_cfg is not None else None
    depth_vis_min_m = float(OmegaConf.select(cfg, "eval.vis_magma_min_m", default=2.0))
    depth_vis_max_m = float(OmegaConf.select(cfg, "eval.vis_magma_max_m", default=100.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "_one_frame_split.txt"
    split_path.write_text(
        f"event3d/DSEC/train/nbin_3_encoding/{args.seq}/{args.event_frame_id:010d}.tif "
        f"{pseudo_path}\n"
    )

    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    loader = build_one_frame_loader(cfg, args.base_data_dir, split_path, depth_transform)
    batch = next(iter(loader))
    dataset = loader.dataset

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    trainer = build_trainer_for_checkpoint(
        cfg,
        checkpoint,
        loader,
        device,
        args.base_ckpt_dir,
        depth_transform,
        args.out_dir,
    )

    depth_gt = batch["depth_raw_linear"].squeeze().numpy()
    pseudo_depth = batch["pseudo_depth_linear"].squeeze().numpy()
    valid_mask = batch["valid_mask_raw"].squeeze().numpy()
    event_rgb = np.moveaxis(batch["rgb_int"].detach().cpu().numpy().squeeze(), 0, -1)
    rgb_u8 = load_left_rgb(args.base_data_dir, args.seq, args.event_frame_id, crop_hw)
    prediction = infer_marigold_metric(trainer, batch, dataset, apply_calibration=True)
    vis_panels = get_visualization_panels(
        event_rgb,
        depth_gt,
        prediction,
        pseudo_depth=pseudo_depth,
        valid_mask=valid_mask,
        depth_vis_min_m=depth_vis_min_m,
        depth_vis_max_m=depth_vis_max_m,
    )

    panels = {
        "event.svg": event_rgb,
        "rgb.svg": rgb_u8,
        "gt_depth.svg": inverse_log_magma_numpy(depth_gt, depth_vis_min_m, depth_vis_max_m),
        "metric_aligned_pseudolabel_da2.svg": inverse_log_magma_numpy(
            pseudo_depth, depth_vis_min_m, depth_vis_max_m
        ),
        "prediction_valid.svg": vis_panels["prediction_valid"],
        "prediction_with_calibration_head.svg": inverse_log_magma_numpy(
            prediction, depth_vis_min_m, depth_vis_max_m
        ),
    }
    for name, panel in panels.items():
        save_rgb_svg(str(args.out_dir / name), panel, dpi=args.svg_dpi)

    (args.out_dir / "metadata.txt").write_text(
        "\n".join(
            [
                f"checkpoint={checkpoint}",
                f"sequence={args.seq}",
                f"event_frame={args.event_frame_id:010d}",
                f"rgb_pseudo_frame={pseudo_frame_id:06d}",
                f"pseudo_label={pseudo_path}",
                f"crop_hw={crop_hw}",
                "calibration_applied=true",
            ]
        )
        + "\n"
    )
    print(f"Saved {len(panels)} SVG panels to {args.out_dir}")


if __name__ == "__main__":
    main()
