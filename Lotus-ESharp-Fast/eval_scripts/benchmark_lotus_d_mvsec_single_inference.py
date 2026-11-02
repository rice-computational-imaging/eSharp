#!/usr/bin/env python3
"""Benchmark one calibrated Lotus-D inference on an MVSEC old-nbin3 input."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import train_diffused_lotus_d_calibration as lotus_d
from benchmark_lotus_d_dsec_single_inference import load_models
from diffused_lotus.marigold_data import _import_marigold, load_marigold_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one calibrated Lotus-D MVSEC inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/path/to/runs/diffused_lotus_d_mvsec_stage3/config.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/path/to/runs/diffused_lotus_d_mvsec_stage3/"
            "checkpoints/step-00019500"
        ),
    )
    parser.add_argument("--seq", default="mvsec_outdoor_night1")
    parser.add_argument("--frame-id", type=int, default=1352)
    parser.add_argument(
        "--pseudo-root",
        default="event3d/MVSEC_regen/pesudo_depth_quantile_shift",
        help="Pseudo-depth root relative to --base-data-dir.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/user/code/paper_assets/lotus_d_mvsec_single_inference_benchmark"),
    )
    return parser.parse_args()


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_one_frame_loader(
    marigold_root: Path,
    marigold_cfg,
    base_data_dir: Path,
    split_file: Path,
    depth_transform,
) -> DataLoader:
    dataset_mode, get_dataset, _, _ = _import_marigold(marigold_root)
    dataset_cfg = OmegaConf.create(
        {
            "name": "mvsec_gt",
            "disp_name": "mvsec_pipeline_one_frame",
            "dir": ".",
            "filenames": str(split_file),
        }
    )
    event_input_mode = OmegaConf.select(
        marigold_cfg, "dataset.train.event_input_mode", default="tenacode"
    )
    dataset = get_dataset(
        dataset_cfg,
        base_data_dir=str(base_data_dir),
        mode=dataset_mode.EVAL,
        depth_transform=depth_transform,
        augmentation_args=marigold_cfg.augmentation,
        event_input_mode=event_input_mode,
    )
    return DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    checkpoint = args.checkpoint.resolve()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    marigold_root = Path(str(cfg.data.marigold_root)).resolve()
    base_data_dir = Path(
        os.environ.get("BASE_DATA_DIR", str(cfg.data.base_data_dir))
    ).resolve()
    marigold_cfg = load_marigold_config(marigold_root, cfg.data.marigold_config)
    _, _, _, get_depth_normalizer = _import_marigold(marigold_root)
    depth_transform = get_depth_normalizer(marigold_cfg.depth_normalization)

    split_path = args.out_dir / "_benchmark_one_frame_split.txt"
    split_path.write_text(
        f"event3d/MVSEC_regen/nbin3/{args.seq}/{args.frame_id:010d}.tif "
        f"{args.pseudo_root}/{args.seq}/frame_{args.frame_id:010d}_pred_depth.npy\n"
    )
    loader = build_one_frame_loader(
        marigold_root, marigold_cfg, base_data_dir, split_path, depth_transform
    )
    batch = next(iter(loader))
    vae, unet, calibration_head, text_embedding, weight_dtype = load_models(
        cfg, checkpoint, device
    )
    lotus_d._CALIBRATION_HEAD = calibration_head

    def infer(seed: int) -> torch.Tensor:
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

    with torch.inference_mode():
        for warmup_index in range(max(args.warmup, 0)):
            infer(int(cfg.validation.seed) + warmup_index)
        synchronize_if_cuda(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
        else:
            memory_before = {"allocated_bytes": None, "reserved_bytes": None}

        times_s = []
        for repeat_index in range(max(args.repeat, 1)):
            synchronize_if_cuda(device)
            start = time.perf_counter()
            prediction = infer(int(cfg.validation.seed) + repeat_index)
            synchronize_if_cuda(device)
            times_s.append(time.perf_counter() - start)

    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_allocated = peak_reserved = device_name = None
    payload = {
        "model": "Lotus-D calibrated",
        "seq": args.seq,
        "frame_id": f"{args.frame_id:010d}",
        "checkpoint": str(checkpoint),
        "config": str(args.config.resolve()),
        "device": str(device),
        "cuda_device_name": device_name,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "times_s": times_s,
        "mean_time_s": sum(times_s) / len(times_s),
        "min_time_s": min(times_s),
        "max_time_s": max(times_s),
        "prediction_shape": list(prediction.shape),
        "cuda_memory_before": memory_before,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_allocated_gb": peak_allocated / 1024**3 if peak_allocated else None,
        "cuda_peak_reserved_gb": peak_reserved / 1024**3 if peak_reserved else None,
    }
    output_path = args.out_dir / "inference_benchmark.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved benchmark to {output_path}")
    print(f"mean_time_s: {payload['mean_time_s']:.4f}")
    if payload["cuda_peak_allocated_gb"] is not None:
        print(f"cuda_peak_allocated_gb: {payload['cuda_peak_allocated_gb']:.3f}")
        print(f"cuda_peak_reserved_gb: {payload['cuda_peak_reserved_gb']:.3f}")


if __name__ == "__main__":
    main()
