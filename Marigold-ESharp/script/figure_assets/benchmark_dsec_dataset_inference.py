#!/usr/bin/env python3
"""Benchmark calibrated Marigold latency over cached DSEC old-nbin3 inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

from generate_dsec_pipeline_svgs import (
    build_one_frame_loader,
    build_trainer_for_checkpoint,
    get_depth_normalizer,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark calibrated Marigold over DSEC validation examples."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/path/to/data/marigold/26_07_10-11_33_00-"
            "train_marigold_dsec_old_nbin_calib_joint_calib_crop/checkpoint/latest"
        ),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=REPO_ROOT / "data_split/dsec_lidar_aligned/gt_val.txt",
    )
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path(os.environ.get("BASE_DATA_DIR", "/path/to/data")),
    )
    parser.add_argument(
        "--base-ckpt-dir",
        type=Path,
        default=Path(os.environ.get("BASE_CKPT_DIR", "/path/to/models")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/user/code/paper_assets/dsec_dataset_inference_benchmark/marigold"),
    )
    return parser.parse_args()


def select_split(source: Path, count: int, seed: int, destination: Path) -> list[str]:
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No entries in {source}")
    selected = random.Random(seed).sample(lines, k=min(count, len(lines)))
    destination.write_text("\n".join(selected) + "\n")
    return selected


@torch.inference_mode()
def infer_metric(trainer, batch: dict, dataset, apply_calibration: bool) -> np.ndarray:
    # Keep the input on GPU before timing; visualization and data loading are excluded.
    rgb_int = batch["rgb_int"]
    if isinstance(rgb_int, torch.Tensor):
        rgb_int = rgb_int.to(trainer.device, non_blocking=True)
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
        color_map=None,
        show_progress_bar=False,
        resample_method=trainer.cfg.validation.resample_method,
    )
    pred = trainer._pred_to_metric_for_eval(pipe_out.depth_np, depth_raw, valid_mask)
    if apply_calibration and trainer.use_calibration_head and trainer.calibration_apply_in_validation:
        pred = trainer._apply_calibration_head_metric(pred)
    return np.clip(pred, 1e-6, dataset.max_depth)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint.resolve()
    cfg = OmegaConf.load(checkpoint.parent.parent / "config.yaml")
    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    selected = select_split(args.split, args.num_frames, args.seed, args.out_dir / "selected_split.txt")
    loader = build_one_frame_loader(cfg, args.base_data_dir, args.out_dir / "selected_split.txt", depth_transform)
    trainer = build_trainer_for_checkpoint(
        cfg, checkpoint, loader, device, args.base_ckpt_dir, depth_transform, args.out_dir
    )
    trainer.model.to(device)
    trainer.model.unet.eval()
    trainer.model.vae.eval()
    trainer.model.text_encoder.eval()
    if trainer.use_calibration_head:
        trainer.calibration_head.eval()
    apply_calibration = bool(trainer.use_calibration_head and trainer.calibration_apply_in_validation)

    iterator = iter(loader)
    first_batch = next(iterator)
    for _ in range(max(args.warmup, 0)):
        infer_metric(trainer, first_batch, loader.dataset, apply_calibration)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    else:
        memory_before = {"allocated_bytes": None, "reserved_bytes": None}

    timings: list[float] = []
    for index, batch in enumerate([first_batch, *iterator]):
        synchronize(device)
        start = time.perf_counter()
        infer_metric(trainer, batch, loader.dataset, apply_calibration)
        synchronize(device)
        timings.append(time.perf_counter() - start)
        print(f"{index + 1:>3}/{len(selected)}: {timings[-1]:.4f} s")

    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_allocated = peak_reserved = device_name = None
    values = np.asarray(timings)
    summary = {
        "model": "Marigold calibrated",
        "checkpoint": str(checkpoint),
        "split": str(args.split.resolve()),
        "num_frames": len(timings),
        "batch_size": 1,
        "timed_region": "prepared input through metric depth prediction; excludes disk loading and visualization",
        "warmup": args.warmup,
        "device": str(device),
        "cuda_device_name": device_name,
        "denoising_steps": int(cfg.validation.denoising_steps),
        "ensemble_size": int(cfg.validation.ensemble_size),
        "mean_time_s": float(values.mean()),
        "median_time_s": float(np.median(values)),
        "p95_time_s": float(np.percentile(values, 95)),
        "min_time_s": float(values.min()),
        "max_time_s": float(values.max()),
        "cuda_memory_before": memory_before,
        "cuda_peak_allocated_gb": peak_allocated / 1024**3 if peak_allocated is not None else None,
        "cuda_peak_reserved_gb": peak_reserved / 1024**3 if peak_reserved is not None else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out_dir / "per_frame_times.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "split_entry", "time_s"])
        writer.writerows((index, selected[index], value) for index, value in enumerate(timings))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
