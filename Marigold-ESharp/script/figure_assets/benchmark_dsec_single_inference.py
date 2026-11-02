#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from generate_dsec_pipeline_svgs import (
    build_one_frame_loader,
    build_trainer_for_checkpoint,
    get_depth_normalizer,
    infer_marigold_metric,
)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one DSEC Marigold inference.")
    parser.add_argument("--seq", default="zurich_city_09_d")
    parser.add_argument("--frame-id", type=int, default=50, help="Old nbin3 event frame id.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/path/to/repository/Marigold/output/train_marigold_dsec_old_nbin_calib_joint_/checkpoint/latest"),
    )
    parser.add_argument("--base-data-dir", type=Path, default=Path(os.environ.get("BASE_DATA_DIR", "/path/to/data")))
    parser.add_argument(
        "--base-ckpt-dir",
        type=Path,
        default=Path(os.environ.get("BASE_CKPT_DIR", "/path/to/models")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/user/code/paper_assets/dsec_single_inference_benchmark"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cfg = OmegaConf.load(args.checkpoint.parent.parent / "config.yaml")
    depth_transform = get_depth_normalizer(cfg.depth_normalization)

    disparity_frame = args.frame_id * 2
    split_path = args.out_dir / "_benchmark_one_frame_split.txt"
    split_path.write_text(
        f"event3d/DSEC/train/nbin_3_encoding/{args.seq}/{args.frame_id:010d}.tif "
        f"dsec_generative_label_event_disparity_affine/{args.seq}/frame{disparity_frame:06d}_pred_depth.npy\n"
    )

    loader = build_one_frame_loader(cfg, args.base_data_dir, split_path, depth_transform)
    batch = next(iter(loader))
    dataset = loader.dataset
    trainer = build_trainer_for_checkpoint(
        cfg, args.checkpoint, loader, device, args.base_ckpt_dir, depth_transform, args.out_dir
    )
    apply_calibration = (not args.no_calibration)

    with torch.no_grad():
        for _ in range(max(args.warmup, 0)):
            _ = infer_marigold_metric(trainer, batch, dataset, apply_calibration=apply_calibration)
        synchronize_if_cuda(device)

        timings = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
        else:
            memory_before = {"allocated_bytes": None, "reserved_bytes": None}

        for _ in range(max(args.repeat, 1)):
            synchronize_if_cuda(device)
            start = time.perf_counter()
            _ = infer_marigold_metric(trainer, batch, dataset, apply_calibration=apply_calibration)
            synchronize_if_cuda(device)
            timings.append(time.perf_counter() - start)

        if device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            device_name = torch.cuda.get_device_name(device)
        else:
            peak_allocated = None
            peak_reserved = None
            device_name = None

    payload = {
        "seq": args.seq,
        "event_frame_id": f"{args.frame_id:010d}",
        "disparity_rgb_frame_id": f"{disparity_frame:06d}",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "cuda_device_name": device_name,
        "apply_calibration": apply_calibration,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "denoising_steps": int(cfg.validation.denoising_steps),
        "ensemble_size": int(cfg.validation.ensemble_size),
        "processing_res": cfg.validation.processing_res,
        "match_input_res": bool(cfg.validation.match_input_res),
        "times_s": timings,
        "mean_time_s": sum(timings) / len(timings),
        "min_time_s": min(timings),
        "max_time_s": max(timings),
        "cuda_memory_before": memory_before,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_allocated_gb": (peak_allocated / 1024**3) if peak_allocated is not None else None,
        "cuda_peak_reserved_gb": (peak_reserved / 1024**3) if peak_reserved is not None else None,
    }

    out_path = args.out_dir / "inference_benchmark.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved benchmark to {out_path}")
    print(f"mean_time_s: {payload['mean_time_s']:.4f}")
    if payload["cuda_peak_allocated_gb"] is not None:
        print(f"cuda_peak_allocated_gb: {payload['cuda_peak_allocated_gb']:.3f}")
        print(f"cuda_peak_reserved_gb: {payload['cuda_peak_reserved_gb']:.3f}")


if __name__ == "__main__":
    main()
