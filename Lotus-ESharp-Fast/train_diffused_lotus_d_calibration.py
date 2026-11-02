#!/usr/bin/env python
"""Fine-tune Lotus-D on DiffusED DSEC event/depth pairs.

This is intentionally separate from ``train_diffused_lotus_g.py``. Lotus-D is
a 4-channel deterministic latent regressor: its U-Net receives only the event
latent and directly predicts either pseudo-depth or the event latent, selected
by the official Lotus task embedding.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPTextModel, CLIPTokenizer
from diffused_lotus.calibration import DepthCalibrationHead, calibration_loss

from diffused_lotus.marigold_data import (
    build_train_dataloader,
    build_validation_dataloaders,
    build_visualization_dataloaders,
)
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

from train_diffused_lotus_g import (
    _empty_text_embedding,
    _latest_checkpoint,
    _masked_latent_loss,
    _prepare_batch,
    _save_checkpoint,
    _task_embedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Lotus-D on DiffusED event/depth pairs without modifying Marigold or Lotus-G."
    )
    parser.add_argument("--config", required=True, help="Lotus-D experiment YAML.")
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


def _validate_model_contract(unet: UNet2DConditionModel) -> None:
    if unet.config.in_channels != 4:
        raise ValueError(f"Lotus-D requires a 4-channel UNet, got {unet.config.in_channels}.")
    if unet.config.out_channels != 4:
        raise ValueError(f"Lotus-D requires a 4-channel latent output, got {unet.config.out_channels}.")
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


def _compute_loss(
    batch: dict,
    cfg: DictConfig,
    vae: AutoencoderKL,
    text_embedding: torch.Tensor,
    unet: UNet2DConditionModel,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float], dict[str, tuple[int, ...]]]:
    event, depth_rgb, valid_mask = _prepare_batch(batch, cfg, device, weight_dtype)
    batch_size = event.shape[0]

    # Official Lotus-D: condition on the clean input latent and directly
    # regress the clean annotation/input latent. No forward diffusion noise.
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
    task_embedding = torch.cat(
        [
            _task_embedding(batch_size, annotation=True, device=device),
            _task_embedding(batch_size, annotation=False, device=device),
        ],
        dim=0,
    ).to(dtype=weight_dtype)
    prediction = unet(
        event_latents,
        timesteps,
        encoder_hidden_states=text_embedding.repeat(2 * batch_size, 1, 1),
        class_labels=task_embedding,
        return_dict=False,
    )[0]

    depth_loss = _masked_latent_loss(
        prediction[:batch_size], target_latents[:batch_size], valid_mask
    )
    reconstruction_loss = F.mse_loss(
        prediction[batch_size:].float(), target_latents[batch_size:].float()
    )
    loss = (
        float(cfg.training.depth_loss_weight) * depth_loss
        + float(cfg.training.reconstruction_loss_weight) * reconstruction_loss
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
        "reconstruction_loss": float(reconstruction_loss.detach()),
        "calibration_loss": float(calibration.detach()),
        "calibration_st_loss": float(st_loss.detach()),
        "calibration_metric_loss": float(metric_loss.detach()),
        "calibration_weight": ramp,
    }
    shapes = {
        "event": tuple(event.shape),
        "depth": tuple(depth_rgb.shape),
        "model_input": tuple(event_latents.shape),
        "prediction": tuple(prediction.shape),
    }
    return loss, values, shapes


@torch.no_grad()
def _predict_metric_depth(
    batch: dict,
    cfg: DictConfig,
    vae: AutoencoderKL,
    text_embedding: torch.Tensor,
    unet: UNet2DConditionModel,
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
    timestep = torch.full(
        (event.shape[0],),
        int(cfg.training.timestep),
        device=device,
        dtype=torch.long,
    )
    task_embedding = _task_embedding(
        event.shape[0], annotation=True, device=device
    ).to(dtype=weight_dtype)
    prediction = unet(
        event_latent,
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
                min=float(loader.dataset.min_depth), max=float(loader.dataset.max_depth)
            )
            for name, function in zip(metric_names, metric_functions):
                value = float(function(prediction, target, valid_mask))
                if math.isfinite(value):
                    values[name].append(value)

        dataset_name = str(loader.dataset.disp_name)
        for name in metric_names:
            totals[f"{dataset_name}/{name}"] = (
                float("nan") if not values[name] else sum(values[name]) / len(values[name])
            )
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
    visualization_loaders,
    depth_transform,
    weight_dtype: torch.dtype,
    output_dir: Path,
    global_step: int,
    report_to: str | None,
) -> None:
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
    vis_root = output_dir / "visualization" / f"step-{global_step:08d}"
    wandb_images = {}
    for loader in visualization_loaders:
        dataset_name = str(loader.dataset.disp_name)
        dataset_dir = vis_root / dataset_name
        images = []
        for sample_index, batch in enumerate(loader):
            if sample_index >= int(cfg.visualization.max_samples):
                break
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(int(cfg.validation.seed) + sample_index)
            prediction = _predict_metric_depth(
                batch=batch,
                cfg=cfg,
                vae=vae,
                text_embedding=text_embedding,
                unet=unet,
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
            svg_name = f"{Path(rel_path).with_suffix('').as_posix().replace('/', '_')}.svg"
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


def main() -> None:
    args = parse_args()
    if args.validate_only and args.resume is None:
        raise ValueError(
            "Calibrated evaluation requires --resume to load the trained calibration head. "
            "For a Stage-I model without a head, use train_diffused_lotus_d.py."
        )
    cfg = OmegaConf.load(args.config)
    if args.validation_max_batches is not None:
        cfg.validation.max_batches = args.validation_max_batches
    output_dir = Path(str(cfg.output_dir)).resolve()

    if args.smoke_test or args.validate_only:
        report_to, project_config = None, None
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
    parameter_groups = [
        {"params": unet.parameters(), "lr": float(cfg.training.learning_rate)},
        {
            "params": calibration_head.parameters(),
            "lr": float(cfg.training.learning_rate) * float(cfg.calibration.head_lr_scale),
        },
    ]
    max_steps = int(cfg.training.max_steps)
    optimizer_name = str(OmegaConf.select(cfg, "training.optimizer", default="adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=float(cfg.training.learning_rate),
            betas=(float(cfg.training.adam_beta1), float(cfg.training.adam_beta2)),
        )
        lr_scheduler = _marigold_scheduler(
            optimizer,
            total_iter=int(cfg.training.lr_scheduler.total_iter),
            final_ratio=float(cfg.training.lr_scheduler.final_ratio),
            warmup_steps=int(cfg.training.lr_scheduler.warmup_steps),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=float(cfg.training.learning_rate),
            betas=(float(cfg.training.adam_beta1), float(cfg.training.adam_beta2)),
            weight_decay=float(cfg.training.weight_decay),
            eps=float(cfg.training.adam_epsilon),
        )
        lr_scheduler = get_scheduler(
            str(cfg.training.lr_scheduler),
            optimizer=optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps),
            num_training_steps=max_steps,
        )
    else:
        raise ValueError(f"Unsupported Stage-II optimizer: {optimizer_name}")
    unet, calibration_head, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, calibration_head, optimizer, train_loader, lr_scheduler
    )
    accelerator.register_for_checkpointing(calibration_head)
    _CALIBRATION_HEAD = calibration_head
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    with torch.no_grad():
        text_embedding = _empty_text_embedding(tokenizer, text_encoder, accelerator.device)

    if args.smoke_test:
        batch = next(iter(train_loader))
        with torch.no_grad(), accelerator.autocast():
            _, values, shapes = _compute_loss(
                batch, cfg, vae, text_embedding, unet, accelerator.device, weight_dtype
            )
        accelerator.print("Lotus-D smoke test passed")
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
            accelerator, cfg, vae, text_embedding, unet, validation_loaders,
            depth_transform, metric_names, metric_functions, weight_dtype,
        )
        accelerator.print(json.dumps(metrics, indent=2))
        eval_root = output_dir / "evaluation"
        if accelerator.is_main_process:
            eval_root.mkdir(parents=True, exist_ok=True)
            (eval_root / f"metrics-step-{global_step:08d}.json").write_text(json.dumps({"step": global_step, **metrics}, indent=2) + "\n")
        accelerator.wait_for_everyone()
        _save_visualizations(
            accelerator, cfg, vae, text_embedding, unet, visualization_loaders,
            depth_transform, weight_dtype, eval_root, global_step, report_to,
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
                        batch, cfg, vae, text_embedding, unet, accelerator.device, weight_dtype
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), float(cfg.training.max_grad_norm))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            _GLOBAL_STEP = global_step
            values["learning_rate"] = float(lr_scheduler.get_last_lr()[0])
            if report_to is not None:
                accelerator.log({f"train/{key}": value for key, value in values.items()}, step=global_step)
            if global_step % int(cfg.training.checkpoint_every) == 0:
                _save_checkpoint(accelerator, output_dir, global_step, cfg)
            if global_step % int(cfg.validation.every) == 0:
                accelerator.print(f"Running validation at step {global_step}...")
                metrics = _run_validation(
                    accelerator, cfg, vae, text_embedding, unet, validation_loaders,
                    depth_transform, metric_names, metric_functions, weight_dtype,
                )
                accelerator.print(
                    f"Step {global_step}. Validation metrics: " + json.dumps(metrics, sort_keys=True)
                )
                if report_to is not None:
                    accelerator.log({f"validation/{key}": value for key, value in metrics.items()}, step=global_step)
                if accelerator.is_main_process:
                    with (output_dir / "validation_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": global_step, **metrics}) + "\n")
                if global_step % int(cfg.visualization.every) == 0:
                    accelerator.print(f"Saving visualizations at step {global_step}...")
                    _save_visualizations(
                        accelerator, cfg, vae, text_embedding, unet, visualization_loaders,
                        depth_transform, weight_dtype, output_dir, global_step, report_to,
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
