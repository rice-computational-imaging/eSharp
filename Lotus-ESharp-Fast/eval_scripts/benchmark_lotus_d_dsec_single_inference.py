#!/usr/bin/env python3
"""Benchmark one calibrated Lotus-D inference on the DSEC old-nbin3 input."""

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
from diffusers import AutoencoderKL, UNet2DConditionModel
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

import train_diffused_lotus_d_calibration as lotus_d
from diffused_lotus.calibration import DepthCalibrationHead
from diffused_lotus.marigold_data import _import_marigold, load_marigold_config
from train_diffused_lotus_g import _empty_text_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one calibrated Lotus-D DSEC inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/template_lotus_d_dsec_stage2_calibration.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/path/to/runs/diffused_lotus_d_dsec_stage2_calibration/"
            "checkpoints/step-00012000"
        ),
        help="Stage-II checkpoint containing model.safetensors and model_1.safetensors.",
    )
    parser.add_argument("--seq", default="zurich_city_09_d")
    parser.add_argument("--frame-id", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/path/to/user/code/paper_assets/lotus_d_dsec_single_inference_benchmark"),
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
            "name": "dsec_event_disparity",
            "disp_name": "dsec_pipeline_one_frame",
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


def load_models(cfg, checkpoint: Path, device: torch.device):
    model_path = Path(str(cfg.model.pretrained_path)).resolve()
    unet_state = checkpoint / "model.safetensors"
    head_state = checkpoint / "model_1.safetensors"
    if not unet_state.is_file() or not head_state.is_file():
        raise FileNotFoundError(
            "Expected model.safetensors and model_1.safetensors under "
            f"{checkpoint}"
        )

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    lotus_d._validate_model_contract(unet)
    unet.load_state_dict(load_safetensors(str(unet_state)), strict=True)

    calibration_head = DepthCalibrationHead(
        int(cfg.calibration.hidden_dim), float(cfg.calibration.t_max_m)
    )
    calibration_head.load_state_dict(load_safetensors(str(head_state)), strict=True)

    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae.requires_grad_(False).eval().to(device=device, dtype=weight_dtype)
    text_encoder.requires_grad_(False).eval().to(device=device, dtype=weight_dtype)
    unet.requires_grad_(False).eval().to(device=device, dtype=weight_dtype)
    calibration_head.requires_grad_(False).eval().to(device=device)
    with torch.inference_mode():
        text_embedding = _empty_text_embedding(tokenizer, text_encoder, device)
    return vae, unet, calibration_head, text_embedding, weight_dtype


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
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

    disparity_frame = args.frame_id * 2
    split_path = args.out_dir / "_benchmark_one_frame_split.txt"
    split_path.write_text(
        f"event3d/DSEC/train/nbin_3_encoding/{args.seq}/{args.frame_id:010d}.tif "
        f"dsec_generative_label_event_disparity_affine/{args.seq}/"
        f"frame{disparity_frame:06d}_pred_depth.npy\n"
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
        "event_frame_id": f"{args.frame_id:010d}",
        "disparity_rgb_frame_id": f"{disparity_frame:06d}",
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
