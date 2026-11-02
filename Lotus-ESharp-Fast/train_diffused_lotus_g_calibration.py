#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPTextModel, CLIPTokenizer
from diffused_lotus.calibration import DepthCalibrationHead, calibration_loss

_CALIBRATION_HEAD = None
_DEPTH_TRANSFORM = None
_GLOBAL_STEP = 0


def _marigold_scheduler(optimizer, total_iter: int, final_ratio: float, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_iter:
            return final_ratio
        return math.exp((step - warmup_steps) / max(1, total_iter - warmup_steps) * math.log(final_ratio))
    return LambdaLR(optimizer, lr_lambda=factor)

from diffused_lotus.marigold_data import (
    build_train_dataloader,
    build_validation_dataloaders,
    build_visualization_dataloaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Lotus-G on DiffusED event/depth pairs without modifying Marigold."
    )
    parser.add_argument("--config", required=True, help="Lotus experiment YAML.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Accelerate checkpoint directory, or 'latest' under output_dir/checkpoints.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one forward/loss pass without an optimizer update or file output.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation plus Marigold-style SVG export once, optionally after loading --resume, then exit.",
    )
    parser.add_argument(
        "--validation-max-batches",
        type=int,
        default=None,
        help="Temporarily cap validation batches without editing the experiment YAML.",
    )
    return parser.parse_args()


def _task_embedding(batch_size: int, annotation: bool, device: torch.device) -> torch.Tensor:
    values = [1.0, 0.0] if annotation else [0.0, 1.0]
    base = torch.tensor(values, device=device).unsqueeze(0)
    return torch.cat([torch.sin(base), torch.cos(base)], dim=-1).repeat(batch_size, 1)


def _masked_latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    invalid = ~valid_mask.bool()
    valid_down = ~F.max_pool2d(invalid.float(), kernel_size=8, stride=8).bool()
    valid_down = valid_down.expand(-1, prediction.shape[1], -1, -1)
    if not torch.any(valid_down):
        raise RuntimeError("Batch has no fully valid 8x8 latent cells for depth supervision.")
    return F.mse_loss(prediction[valid_down].float(), target[valid_down].float())


def _empty_text_embedding(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    text_inputs = tokenizer(
        "",
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]


def _validate_model_contract(unet: UNet2DConditionModel) -> None:
    if unet.config.in_channels != 8:
        raise ValueError(
            f"Lotus-G requires an 8-channel UNet, got {unet.config.in_channels}."
        )
    if unet.config.out_channels != 4:
        raise ValueError(
            f"Lotus-G requires a 4-channel latent output, got {unet.config.out_channels}."
        )
    if unet.config.class_embed_type != "projection":
        raise ValueError(
            "Checkpoint is missing Lotus task conditioning "
            f"(class_embed_type={unet.config.class_embed_type!r})."
        )
    if unet.config.projection_class_embeddings_input_dim != 4:
        raise ValueError(
            "Lotus task embedding must have dimension 4, got "
            f"{unet.config.projection_class_embeddings_input_dim}."
        )


def _prepare_batch(
    batch: dict,
    cfg: DictConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_key = str(cfg.data.input_key)
    depth_key = str(cfg.data.depth_key)
    mask_key = str(cfg.data.mask_key)
    event = batch[input_key].to(device=device, dtype=dtype)
    depth = batch[depth_key].to(device=device, dtype=dtype)
    valid_mask = batch[mask_key].to(device=device).bool()

    if event.ndim != 4 or event.shape[1] != 3:
        raise ValueError(f"Expected event tensor [B,3,H,W], got {tuple(event.shape)}")
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth tensor [B,1,H,W], got {tuple(depth.shape)}")
    if valid_mask.shape != depth.shape:
        raise ValueError(
            f"Depth/mask shapes differ: depth={tuple(depth.shape)}, mask={tuple(valid_mask.shape)}"
        )
    if event.shape[-2:] != depth.shape[-2:]:
        raise ValueError(
            f"Event/depth resolutions differ: event={tuple(event.shape)}, depth={tuple(depth.shape)}"
        )
    if event.shape[-2] % 8 or event.shape[-1] % 8:
        raise ValueError(f"Input resolution must be divisible by 8, got {event.shape[-2:]}")
    if not torch.isfinite(event).all() or not torch.isfinite(depth).all():
        raise ValueError("Event or normalized pseudo-depth contains NaN/Inf.")

    depth_rgb = depth.repeat(1, 3, 1, 1)
    return event, depth_rgb, valid_mask


def _compute_loss(
    batch: dict,
    cfg: DictConfig,
    vae: AutoencoderKL,
    text_embedding: torch.Tensor,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float], dict[str, tuple[int, ...]]]:
    event, depth_rgb, valid_mask = _prepare_batch(batch, cfg, device, weight_dtype)
    batch_size = event.shape[0]

    with torch.no_grad():
        paired_events = torch.cat([event, event], dim=0)
        paired_targets = torch.cat([depth_rgb, event], dim=0)
        event_latents = vae.encode(paired_events).latent_dist.sample()
        target_latents = vae.encode(paired_targets).latent_dist.sample()
        event_latents = event_latents * vae.config.scaling_factor
        target_latents = target_latents * vae.config.scaling_factor

    timesteps = torch.full(
        (2 * batch_size,),
        int(cfg.training.timestep),
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target_latents)
    noisy_targets = noise_scheduler.add_noise(target_latents, noise, timesteps)
    model_input = torch.cat([event_latents, noisy_targets], dim=1)

    prompt_embedding = text_embedding.repeat(2 * batch_size, 1, 1)
    task_embedding = torch.cat(
        [
            _task_embedding(batch_size, annotation=True, device=device),
            _task_embedding(batch_size, annotation=False, device=device),
        ],
        dim=0,
    ).to(dtype=weight_dtype)

    prediction = unet(
        model_input,
        timesteps,
        encoder_hidden_states=prompt_embedding,
        class_labels=task_embedding,
        return_dict=False,
    )[0]

    depth_loss = _masked_latent_loss(
        prediction[:batch_size], target_latents[:batch_size], valid_mask
    )
    identity_loss = F.mse_loss(
        prediction[batch_size:].float(), target_latents[batch_size:].float()
    )
    loss = (
        float(cfg.training.depth_loss_weight) * depth_loss
        + float(cfg.training.identity_loss_weight) * identity_loss
    )
    decoded = vae.decode(prediction[:batch_size] / vae.config.scaling_factor, return_dict=False)[0].mean(dim=1, keepdim=True)
    metric_prediction = _DEPTH_TRANSFORM.denormalize(decoded.float())
    metric_target = batch[str(cfg.data.metric_target_key)].to(device=device, dtype=torch.float32)
    metric_mask = batch[str(cfg.data.metric_mask_key)].to(device=device).bool()
    calibration_input = metric_prediction.detach() if bool(cfg.stage1.freeze_backbone) else metric_prediction
    calibration, _, st_loss, metric_loss = calibration_loss(_CALIBRATION_HEAD, calibration_input, metric_target, metric_mask, float(cfg.calibration.lambda_st), float(cfg.calibration.lambda_t), float(cfg.calibration.lambda_metric))
    ramp = min(1.0, max(0.0, (_GLOBAL_STEP - int(cfg.calibration.warmup_iters)) / max(1, int(cfg.calibration.ramp_iters))))
    loss = loss + ramp * calibration
    values = {
        "loss": float(loss.detach()),
        "depth_loss": float(depth_loss.detach()),
        "identity_loss": float(identity_loss.detach()),
        "calibration_loss": float(calibration.detach()),
        "calibration_st_loss": float(st_loss.detach()),
        "calibration_metric_loss": float(metric_loss.detach()),
        "calibration_weight": ramp,
    }
    shapes = {
        "event": tuple(event.shape),
        "depth": tuple(depth_rgb.shape),
        "model_input": tuple(model_input.shape),
        "prediction": tuple(prediction.shape),
    }
    return loss, values, shapes


def _latest_checkpoint(output_dir: Path) -> Path:
    checkpoint_root = output_dir / "checkpoints"
    checkpoints = sorted(
        checkpoint_root.glob("step-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_root}")
    return checkpoints[-1]


@torch.no_grad()
def _predict_metric_depth(
    batch: dict,
    cfg: DictConfig,
    vae: AutoencoderKL,
    text_embedding: torch.Tensor,
    unet: UNet2DConditionModel,
    inference_scheduler: DDIMScheduler,
    depth_transform,
    device: torch.device,
    weight_dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    event = batch[str(cfg.data.input_key)].to(device=device, dtype=weight_dtype)
    if event.ndim != 4 or event.shape[1] != 3:
        raise ValueError(f"Expected validation event [B,3,H,W], got {tuple(event.shape)}")

    event_latent = vae.encode(event).latent_dist.sample(generator=generator)
    event_latent = event_latent * vae.config.scaling_factor
    noisy_depth = torch.randn(
        event_latent.shape,
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    noisy_depth = noisy_depth * inference_scheduler.init_noise_sigma
    timestep = torch.full(
        (event.shape[0],),
        int(cfg.training.timestep),
        device=device,
        dtype=torch.long,
    )
    noisy_depth_input = inference_scheduler.scale_model_input(noisy_depth, timestep[0])
    model_input = torch.cat([event_latent, noisy_depth_input], dim=1)
    task_embedding = _task_embedding(
        event.shape[0], annotation=True, device=device
    ).to(dtype=weight_dtype)
    prediction = unet(
        model_input,
        timestep,
        encoder_hidden_states=text_embedding.repeat(event.shape[0], 1, 1),
        class_labels=task_embedding,
        return_dict=False,
    )[0]
    decoded = vae.decode(
        prediction.to(dtype=weight_dtype) / vae.config.scaling_factor,
        return_dict=False,
    )[0]
    depth_norm = decoded.mean(dim=1).clamp(-1.0, 1.0).float()
    depth_metric = depth_transform.denormalize(depth_norm)
    if bool(cfg.calibration.apply_in_validation):
        depth_metric_4d = depth_metric.unsqueeze(1)
        scale, shift = _CALIBRATION_HEAD(depth_metric_4d)
        depth_metric = (scale[:, None, None, None] * depth_metric_4d + shift[:, None, None, None]).squeeze(1)
    return depth_metric.clamp_min(1e-6)


@torch.no_grad()
def _run_validation(
    accelerator: Accelerator,
    cfg: DictConfig,
    vae: AutoencoderKL,
    text_embedding: torch.Tensor,
    unet: UNet2DConditionModel,
    inference_scheduler: DDIMScheduler,
    validation_loaders,
    depth_transform,
    metric_names,
    metric_functions,
    weight_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = unet.training
    unet.eval()
    calibration_was_training = _CALIBRATION_HEAD.training
    _CALIBRATION_HEAD.eval()
    totals: dict[str, float] = {}
    max_batches = cfg.validation.max_batches
    for loader in validation_loaders:
        values = {name: [] for name in metric_names}
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(int(cfg.validation.seed) + batch_index)
            prediction = _predict_metric_depth(
                batch=batch,
                cfg=cfg,
                vae=vae,
                text_embedding=text_embedding,
                unet=unet,
                inference_scheduler=inference_scheduler,
                depth_transform=depth_transform,
                device=accelerator.device,
                weight_dtype=weight_dtype,
                generator=generator,
            )
            target = batch["depth_raw_linear"].squeeze(1).to(
                accelerator.device, dtype=torch.float32
            )
            valid_mask = batch["valid_mask_raw"].squeeze(1).to(accelerator.device).bool()
            prediction = prediction.clamp(
                min=float(loader.dataset.min_depth),
                max=float(loader.dataset.max_depth),
            )
            for name, function in zip(metric_names, metric_functions):
                value = float(function(prediction, target, valid_mask))
                if math.isfinite(value):
                    values[name].append(value)

        dataset_name = str(loader.dataset.disp_name)
        for name in metric_names:
            if not values[name]:
                totals[f"{dataset_name}/{name}"] = float("nan")
            else:
                totals[f"{dataset_name}/{name}"] = sum(values[name]) / len(values[name])
    if calibration_was_training:
        _CALIBRATION_HEAD.train()
    if was_training:
        unet.train()
    return totals


@torch.no_grad()
def _save_visualizations(
    accelerator: Accelerator,
    cfg: DictConfig,
    vae: AutoencoderKL,
    text_embedding: torch.Tensor,
    unet: UNet2DConditionModel,
    inference_scheduler: DDIMScheduler,
    visualization_loaders,
    depth_transform,
    weight_dtype: torch.dtype,
    output_dir: Path,
    global_step: int,
    report_to: str | None,
) -> None:
    """Save and optionally log Marigold-style qualitative DSEC panels."""
    if not accelerator.is_main_process:
        return

    from src.util.logging_util import (
        get_visualization_image,
        get_visualization_panels,
        save_magma_colorbar_svg,
        save_rgb_svg,
    )

    was_training = unet.training
    unet.eval()
    calibration_was_training = _CALIBRATION_HEAD.training
    _CALIBRATION_HEAD.eval()
    max_samples = int(cfg.visualization.max_samples)
    vis_root = output_dir / "visualization" / f"step-{global_step:08d}"
    wandb_images = {}
    for loader in visualization_loaders:
        dataset_name = str(loader.dataset.disp_name)
        dataset_dir = vis_root / dataset_name
        images = []
        for sample_index, batch in enumerate(loader):
            if sample_index >= max_samples:
                break
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(int(cfg.validation.seed) + sample_index)
            prediction = _predict_metric_depth(
                batch=batch,
                cfg=cfg,
                vae=vae,
                text_embedding=text_embedding,
                unet=unet,
                inference_scheduler=inference_scheduler,
                depth_transform=depth_transform,
                device=accelerator.device,
                weight_dtype=weight_dtype,
                generator=generator,
            ).squeeze(0).detach().cpu().numpy()
            target = batch["depth_raw_linear"].squeeze().numpy()
            valid_mask = batch["valid_mask_raw"].squeeze().numpy().astype(bool)
            pseudo_depth = batch.get("pseudo_depth_linear")
            if pseudo_depth is not None:
                pseudo_depth = pseudo_depth.squeeze().numpy()
            event = batch["rgb_int"].squeeze(0).permute(1, 2, 0).numpy()
            prediction = prediction.clip(
                min=float(loader.dataset.min_depth), max=float(loader.dataset.max_depth)
            )
            panels = get_visualization_panels(
                event,
                target,
                prediction,
                pseudo_depth=pseudo_depth,
                valid_mask=valid_mask,
                depth_vis_min_m=float(cfg.visualization.depth_vis_min_m),
                depth_vis_max_m=float(cfg.visualization.depth_vis_max_m),
            )
            rel_path = str(batch["rgb_relative_path"][0])
            stem = Path(rel_path).with_suffix("").as_posix().replace("/", "_")
            svg_name = f"{stem}.svg"
            for panel_name, image in panels.items():
                save_rgb_svg(str(dataset_dir / panel_name / svg_name), image)
            save_magma_colorbar_svg(
                str(dataset_dir / "colorbar" / "magma_colorbar.svg"),
                target.shape[0],
                depth_vis_min_m=float(cfg.visualization.depth_vis_min_m),
                depth_vis_max_m=float(cfg.visualization.depth_vis_max_m),
            )
            montage = get_visualization_image(
                event,
                target,
                prediction,
                pseudo_depth=pseudo_depth,
                valid_mask=valid_mask,
                depth_vis_min_m=float(cfg.visualization.depth_vis_min_m),
                depth_vis_max_m=float(cfg.visualization.depth_vis_max_m),
            )
            save_rgb_svg(str(dataset_dir / f"vis_all_{svg_name}"), montage)
            images.append((montage, rel_path))
        wandb_images[dataset_name] = images

    if report_to == "wandb":
        import wandb

        tracker = accelerator.get_tracker("wandb", unwrap=True)
        for dataset_name, images in wandb_images.items():
            tracker.log(
                {
                    f"visualization/{dataset_name}": [
                        wandb.Image(image, caption=caption) for image, caption in images
                    ]
                },
                step=global_step,
            )
    if calibration_was_training:
        _CALIBRATION_HEAD.train()
    if was_training:
        unet.train()


def _save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    global_step: int,
    cfg: DictConfig,
) -> None:
    checkpoint_dir = output_dir / "checkpoints" / f"step-{global_step:08d}"
    accelerator.print(f"Saving checkpoint at step {global_step}: {checkpoint_dir}")
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        state = {"global_step": global_step, "config": OmegaConf.to_container(cfg)}
        with (checkpoint_dir / "training_state.json").open("w") as handle:
            json.dump(state, handle, indent=2)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.validation_max_batches is not None:
        cfg.validation.max_batches = args.validation_max_batches
    output_dir = Path(str(cfg.output_dir)).resolve()

    if args.smoke_test or args.validate_only:
        report_to = None
        project_config = None
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_to = None if str(cfg.logging.report_to) == "none" else str(cfg.logging.report_to)
        project_config = ProjectConfiguration(project_dir=str(output_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        mixed_precision=str(cfg.training.mixed_precision),
        log_with=report_to,
        project_config=project_config,
    )
    set_seed(int(cfg.seed))

    model_path = str(Path(str(cfg.model.pretrained_path)).resolve())
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    stage1_path = Path(str(cfg.stage1.checkpoint))
    if not stage1_path.is_dir():
        raise FileNotFoundError(f"Stage-I checkpoint directory does not exist: {stage1_path}")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    if (stage1_path / "config.json").is_file():
        unet = UNet2DConditionModel.from_pretrained(str(stage1_path))
    else:
        state_path = stage1_path / "model.safetensors"
        if not state_path.is_file():
            raise FileNotFoundError(f"Expected config.json or model.safetensors under {stage1_path}")
        unet.load_state_dict(load_safetensors(str(state_path)), strict=True)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    inference_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    _validate_model_contract(unet)

    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    freeze_backbone = bool(cfg.stage1.freeze_backbone)
    unet.requires_grad_(not freeze_backbone).train()
    if bool(cfg.training.gradient_checkpointing):
        unet.enable_gradient_checkpointing()
    calibration_head = DepthCalibrationHead(int(cfg.calibration.hidden_dim), float(cfg.calibration.t_max_m)).train()

    train_loader, marigold_cfg = build_train_dataloader(
        marigold_root=cfg.data.marigold_root,
        marigold_config=cfg.data.marigold_config,
        base_data_dir=cfg.data.base_data_dir,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.data.num_workers),
        seed=int(cfg.seed),
    )

    global _CALIBRATION_HEAD, _DEPTH_TRANSFORM, _GLOBAL_STEP
    from src.util.depth_transform import get_depth_normalizer
    _CALIBRATION_HEAD = calibration_head
    _DEPTH_TRANSFORM = get_depth_normalizer(marigold_cfg.depth_normalization)
    optimizer = torch.optim.Adam(
        [{"params": unet.parameters(), "lr": float(cfg.training.learning_rate)}, {"params": calibration_head.parameters(), "lr": float(cfg.training.learning_rate) * float(cfg.calibration.head_lr_scale)}],
        lr=float(cfg.training.learning_rate),
        betas=(float(cfg.training.adam_beta1), float(cfg.training.adam_beta2)),
    )
    max_steps = int(cfg.training.max_steps)
    lr_scheduler = _marigold_scheduler(
        optimizer,
        total_iter=int(cfg.training.lr_scheduler.total_iter),
        final_ratio=float(cfg.training.lr_scheduler.final_ratio),
        warmup_steps=int(cfg.training.lr_scheduler.warmup_steps),
    )

    unet, calibration_head, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, calibration_head, optimizer, train_loader, lr_scheduler
    )
    accelerator.register_for_checkpointing(calibration_head)
    _CALIBRATION_HEAD = calibration_head
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    with torch.no_grad():
        text_embedding = _empty_text_embedding(tokenizer, text_encoder, accelerator.device)

    if args.smoke_test:
        batch = next(iter(train_loader))
        with torch.no_grad(), accelerator.autocast():
            _, values, shapes = _compute_loss(
                batch,
                cfg,
                vae,
                text_embedding,
                unet,
                noise_scheduler,
                accelerator.device,
                weight_dtype,
            )
        accelerator.print("Lotus-G smoke test passed")
        accelerator.print(json.dumps({"losses": values, "shapes": shapes}, indent=2))
        return

    validation_loaders, depth_transform, _ = build_validation_dataloaders(
        marigold_root=cfg.data.marigold_root,
        marigold_config=cfg.data.marigold_config,
        base_data_dir=cfg.data.base_data_dir,
        num_workers=int(cfg.data.num_workers),
    )
    visualization_loaders = build_visualization_dataloaders(
        marigold_root=cfg.data.marigold_root,
        marigold_config=cfg.data.marigold_config,
        base_data_dir=cfg.data.base_data_dir,
        num_workers=int(cfg.data.num_workers),
    )
    from src.util.metric import build_eval_metric_callables

    metric_names, metric_functions = build_eval_metric_callables(
        cfg.validation.metrics,
        compare_clip_min_m=float(marigold_cfg.eval.compare_clip_min_m),
        compare_clip_max_m=float(marigold_cfg.eval.compare_clip_max_m),
        compare_ignore_top_rows=int(marigold_cfg.eval.compare_ignore_top_rows),
    )

    if accelerator.is_main_process and not args.validate_only:
        OmegaConf.save(cfg, output_dir / "config.yaml")
        OmegaConf.save(marigold_cfg, output_dir / "resolved_marigold_config.yaml")
    if report_to is not None:
        tracker_kwargs = {"wandb": {"name": str(cfg.logging.run_name)}}
        if cfg.logging.get("entity") is not None:
            tracker_kwargs["wandb"]["entity"] = str(cfg.logging.entity)
        accelerator.init_trackers(
            str(cfg.logging.project),
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs=tracker_kwargs,
        )

    global_step = 0
    if args.resume is not None:
        resume_path = _latest_checkpoint(output_dir) if args.resume == "latest" else Path(args.resume)
        if args.validate_only:
            unet_state = resume_path / "model.safetensors"
            head_state = resume_path / "model_1.safetensors"
            if not unet_state.is_file() or not head_state.is_file():
                raise FileNotFoundError(f"Eval-only needs model.safetensors and model_1.safetensors under {resume_path}")
            accelerator.unwrap_model(unet).load_state_dict(load_safetensors(str(unet_state)), strict=True)
            accelerator.unwrap_model(calibration_head).load_state_dict(load_safetensors(str(head_state)), strict=True)
        else:
            accelerator.load_state(str(resume_path))
        state_path = resume_path / "training_state.json"
        if state_path.is_file():
            global_step = int(json.loads(state_path.read_text())["global_step"])
        accelerator.print(f"Resumed from {resume_path} at step {global_step}")

    if args.validate_only:
        metrics = _run_validation(
            accelerator=accelerator,
            cfg=cfg,
            vae=vae,
            text_embedding=text_embedding,
            unet=unet,
            inference_scheduler=inference_scheduler,
            validation_loaders=validation_loaders,
            depth_transform=depth_transform,
            metric_names=metric_names,
            metric_functions=metric_functions,
            weight_dtype=weight_dtype,
        )
        accelerator.print(json.dumps(metrics, indent=2))
        eval_root = output_dir / "evaluation"
        if accelerator.is_main_process:
            eval_root.mkdir(parents=True, exist_ok=True)
            (eval_root / f"metrics-step-{global_step:08d}.json").write_text(json.dumps({"step": global_step, **metrics}, indent=2) + "\n")
        accelerator.wait_for_everyone()
        _save_visualizations(
            accelerator=accelerator, cfg=cfg, vae=vae, text_embedding=text_embedding,
            unet=unet, inference_scheduler=inference_scheduler,
            visualization_loaders=visualization_loaders, depth_transform=depth_transform,
            weight_dtype=weight_dtype, output_dir=eval_root, global_step=global_step, report_to=report_to,
        )
        accelerator.wait_for_everyone()
        return

    steps_per_epoch = math.ceil(len(train_loader) / int(cfg.training.gradient_accumulation_steps))
    num_epochs = math.ceil(max_steps / max(1, steps_per_epoch))
    for _ in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    loss, values, _ = _compute_loss(
                        batch,
                        cfg,
                        vae,
                        text_embedding,
                        unet,
                        noise_scheduler,
                        accelerator.device,
                        weight_dtype,
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), float(cfg.training.max_grad_norm))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                _GLOBAL_STEP = global_step
                values["learning_rate"] = float(lr_scheduler.get_last_lr()[0])
                # Match Marigold: record every effective iteration, but keep these
                # high-frequency scalars out of the terminal.
                if report_to is not None:
                    accelerator.log(
                        {f"train/{key}": value for key, value in values.items()},
                        step=global_step,
                    )
                if global_step % int(cfg.training.checkpoint_every) == 0:
                    _save_checkpoint(accelerator, output_dir, global_step, cfg)
                if global_step % int(cfg.validation.every) == 0:
                    accelerator.print(f"Running validation at step {global_step}...")
                    metrics = _run_validation(
                        accelerator=accelerator,
                        cfg=cfg,
                        vae=vae,
                        text_embedding=text_embedding,
                        unet=unet,
                        inference_scheduler=inference_scheduler,
                        validation_loaders=validation_loaders,
                        depth_transform=depth_transform,
                        metric_names=metric_names,
                        metric_functions=metric_functions,
                        weight_dtype=weight_dtype,
                    )
                    accelerator.print(
                        f"Step {global_step}. Validation metrics: "
                        + json.dumps(metrics, sort_keys=True)
                    )
                    if report_to is not None:
                        accelerator.log({f"validation/{key}": value for key, value in metrics.items()}, step=global_step)
                    if accelerator.is_main_process:
                        with (output_dir / "validation_metrics.jsonl").open("a") as handle:
                            handle.write(
                                json.dumps({"step": global_step, **metrics}) + "\n"
                            )
                    if global_step % int(cfg.visualization.every) == 0:
                        accelerator.print(f"Saving visualizations at step {global_step}...")
                        _save_visualizations(
                            accelerator=accelerator,
                            cfg=cfg,
                            vae=vae,
                            text_embedding=text_embedding,
                            unet=unet,
                            inference_scheduler=inference_scheduler,
                            visualization_loaders=visualization_loaders,
                            depth_transform=depth_transform,
                            weight_dtype=weight_dtype,
                            output_dir=output_dir,
                            global_step=global_step,
                            report_to=report_to,
                        )
                        accelerator.wait_for_everyone()
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        accelerator.print(f"Training ended. Saving final UNet to: {final_dir / 'unet'}")
        accelerator.unwrap_model(unet).save_pretrained(final_dir / "unet")
        torch.save(accelerator.unwrap_model(calibration_head).state_dict(), final_dir / "calibration_head.pt")
        OmegaConf.save(cfg, final_dir / "config.yaml")
    accelerator.end_training()


if __name__ == "__main__":
    main()
