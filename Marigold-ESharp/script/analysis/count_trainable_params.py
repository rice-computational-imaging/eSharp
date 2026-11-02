#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from marigold.marigold_pipeline import MarigoldPipeline
from src.trainer import get_trainer_cls
from src.util.config_util import recursive_load_config
from src.util.depth_transform import get_depth_normalizer


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    if args.run_dir is not None:
        return args.run_dir / "config.yaml"
    if args.checkpoint is not None:
        return args.checkpoint.parent.parent / "config.yaml"
    raise ValueError("Pass one of --config, --run-dir, or --checkpoint.")


def load_config(config_path: Path) -> OmegaConf:
    if config_path.name == "config.yaml":
        return OmegaConf.load(config_path)
    return recursive_load_config(str(config_path))


def count_params(module: torch.nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "trainable_percent": float(100.0 * trainable / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count total and trainable parameters for a Marigold training stage. "
            "This reconstructs the trainer from config and does not run training or evaluation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path like /path/to/run/checkpoint/latest.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing config.yaml.")
    parser.add_argument("--config", type=Path, default=None, help="Standalone config file to count from.")
    parser.add_argument(
        "--base-ckpt-dir",
        type=Path,
        default=Path(os.environ.get("BASE_CKPT_DIR", "/path/to/models")),
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default="cpu", help="Use cpu by default; cuda is not needed for counting.")
    args = parser.parse_args()

    config_path = resolve_config_path(args)
    cfg = load_config(config_path)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    dummy_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

    model = MarigoldPipeline.from_pretrained(
        str(args.base_ckpt_dir / cfg.model.pretrained_path),
        **(cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}),
    )
    trainer_cls = get_trainer_cls(cfg.trainer.name)

    # The trainer constructor applies the actual trainability rules:
    # VAE/text encoder frozen, UNet optionally frozen, calibration head optionally enabled.
    scratch_root = REPO_ROOT / "output" / "_param_count_unused"
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=dummy_loader,
        device=device,
        base_ckpt_dir=str(args.base_ckpt_dir),
        out_dir_ckpt=str(scratch_root / "checkpoint"),
        out_dir_eval=str(scratch_root / "eval"),
        out_dir_vis=str(scratch_root / "visualization"),
        accumulation_steps=1,
        val_dataloaders=[],
        vis_dataloaders=[],
        depth_transform=depth_transform,
    )

    modules: dict[str, torch.nn.Module] = {
        "unet": trainer.model.unet,
        "vae": trainer.model.vae,
        "text_encoder": trainer.model.text_encoder,
    }
    if getattr(trainer, "use_calibration_head", False):
        modules["calibration_head"] = trainer.calibration_head

    module_counts = {name: count_params(module) for name, module in modules.items()}
    total = sum(item["total"] for item in module_counts.values())
    trainable = sum(item["trainable"] for item in module_counts.values())

    payload = {
        "config": str(config_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "base_ckpt_dir": str(args.base_ckpt_dir),
        "trainer": cfg.trainer.name,
        "calibration_enabled": bool(OmegaConf.select(cfg, "loss.calibration.enabled", default=False)),
        "calibration_freeze_backbone": bool(
            OmegaConf.select(cfg, "loss.calibration.freeze_backbone", default=False)
        ),
        "modules": module_counts,
        "all_counted_modules": {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(total - trainable),
            "trainable_percent": float(100.0 * trainable / total) if total else 0.0,
        },
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
