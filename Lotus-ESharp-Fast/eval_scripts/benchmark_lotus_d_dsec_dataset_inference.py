#!/usr/bin/env python3
"""Benchmark calibrated Lotus-D latency over cached DSEC old-nbin3 inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_lotus_d_dsec_single_inference import build_one_frame_loader, load_models
import train_diffused_lotus_d_calibration as lotus_d
from diffused_lotus.marigold_data import _import_marigold, load_marigold_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark calibrated Lotus-D over DSEC validation examples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/template_lotus_d_dsec_stage2_calibration.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/path/to/runs/diffused_lotus_d_3_stage/"
            "checkpoints/step-00017000"
        ),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("/path/to/repository/Marigold-ESharp/data_split/dsec_lidar_aligned/gt_val.txt"),
    )
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/user/code/paper_assets/dsec_dataset_inference_benchmark/lotus_d"),
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_split(source: Path, count: int, seed: int, destination: Path) -> list[str]:
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No entries in {source}")
    selected = random.Random(seed).sample(lines, k=min(count, len(lines)))
    destination.write_text("\n".join(selected) + "\n")
    return selected


def prepare_batch(batch: dict, key: str, device: torch.device, dtype: torch.dtype) -> dict:
    prepared = dict(batch)
    prepared[key] = batch[key].to(device=device, dtype=dtype, non_blocking=True)
    return prepared


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    marigold_root = Path(str(cfg.data.marigold_root)).resolve()
    base_data_dir = Path(os.environ.get("BASE_DATA_DIR", str(cfg.data.base_data_dir))).resolve()
    marigold_cfg = load_marigold_config(marigold_root, cfg.data.marigold_config)
    _, _, _, get_depth_normalizer = _import_marigold(marigold_root)
    depth_transform = get_depth_normalizer(marigold_cfg.depth_normalization)
    selected = select_split(args.split, args.num_frames, args.seed, args.out_dir / "selected_split.txt")
    loader = build_one_frame_loader(
        marigold_root, marigold_cfg, base_data_dir, args.out_dir / "selected_split.txt", depth_transform
    )
    vae, unet, calibration_head, text_embedding, weight_dtype = load_models(cfg, checkpoint, device)
    lotus_d._CALIBRATION_HEAD = calibration_head

    def infer(batch: dict, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return lotus_d._predict_metric_depth(
            batch=batch,
            cfg=cfg,
            vae=vae,
            text_embedding=text_embedding,
            unet=unet,
            depth_transform=depth_transform,
            device=device,
            weight_dtype=weight_dtype,
            generator=generator,
        )

    input_key = str(cfg.data.input_key)
    batches = [prepare_batch(batch, input_key, device, weight_dtype) for batch in loader]
    for warmup_index in range(max(args.warmup, 0)):
        infer(batches[0], int(cfg.validation.seed) + warmup_index)
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
    for index, batch in enumerate(batches):
        synchronize(device)
        start = time.perf_counter()
        infer(batch, int(cfg.validation.seed) + index)
        synchronize(device)
        timings.append(time.perf_counter() - start)
        print(f"{index + 1:>3}/{len(batches)}: {timings[-1]:.4f} s")

    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_allocated = peak_reserved = device_name = None
    values = np.asarray(timings)
    summary = {
        "model": "Lotus-D calibrated",
        "checkpoint": str(checkpoint),
        "config": str(args.config.resolve()),
        "split": str(args.split.resolve()),
        "num_frames": len(timings),
        "batch_size": 1,
        "timed_region": "prepared GPU event tensor through calibrated metric depth; excludes disk loading, event encoding, and H2D transfer",
        "warmup": args.warmup,
        "device": str(device),
        "cuda_device_name": device_name,
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
