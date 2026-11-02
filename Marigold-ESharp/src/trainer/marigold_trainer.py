# An official reimplemented version of Marigold training script.
# Last modified: 2024-08-16
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import csv
import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
import wandb
import torch.nn.functional as F
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm as original_tqdm  # Import the original tqdm function
# Redefine tqdm to set ascii=True by default
def tqdm(*args, **kwargs):
    return original_tqdm(*args, dynamic_ncols=True, ascii=True, **kwargs)

import imageio.v3 as io

from marigold.marigold_pipeline import MarigoldPipeline, MarigoldDepthOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import (
    tb_logger,
    eval_dic_to_text,
    get_visualization_image,
    get_visualization_panels,
    save_magma_colorbar_svg,
    save_rgb_svg,
)
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square, align_depth_quantile_map
from src.util.calibration import (
    apply_affine,
    build_calibration_head,
    masked_l1_mean,
    solve_affine_least_squares_batch,
)
from src.util.seeding import generate_seed_sequence
from marigold.util.image_util import get_edge_map



class MarigoldTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        depth_transform,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: MarigoldPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        self.depth_transform = depth_transform

        # Adapt input layers
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        self.model.unet.enable_xformers_memory_efficient_attention()

        # Optional post calibration head (predict affine scale/shift).
        self.use_calibration_head = OmegaConf.select(
            self.cfg, "loss.calibration.enabled", default=False
        )
        self.calibration_freeze_backbone = OmegaConf.select(
            self.cfg, "loss.calibration.freeze_backbone", default=False
        )
        self.calibration_head_lr_scale = OmegaConf.select(
            self.cfg, "loss.calibration.head_lr_scale", default=1.0
        )
        self.calibration_lambda_st = OmegaConf.select(
            self.cfg, "loss.calibration.lambda_st", default=1.0
        )
        self.calibration_lambda_t = OmegaConf.select(
            self.cfg, "loss.calibration.lambda_t", default=1.0
        )
        self.calibration_lambda_metric = OmegaConf.select(
            self.cfg, "loss.calibration.lambda_metric", default=0.1
        )
        self.calibration_warmup_iters = OmegaConf.select(
            self.cfg, "loss.calibration.warmup_iters", default=500
        )
        self.calibration_ramp_iters = OmegaConf.select(
            self.cfg, "loss.calibration.ramp_iters", default=2000
        )
        self.calibration_histogram_period = OmegaConf.select(
            self.cfg, "loss.calibration.histogram_period", default=200
        )
        self.calibration_apply_in_validation = OmegaConf.select(
            self.cfg, "loss.calibration.apply_in_validation", default=True
        )
        self.calibration_head = None
        if self.use_calibration_head:
            hidden_dim = OmegaConf.select(
                self.cfg, "loss.calibration.hidden_dim", default=32
            )
            t_max_m = OmegaConf.select(
                self.cfg, "loss.calibration.t_max_m", default=20.0
            )
            self.calibration_head = build_calibration_head(
                hidden_dim=hidden_dim,
                t_max_m=t_max_m,
            ).to(device)
            logging.info(
                "Calibration head enabled: freeze_backbone=%s head_lr_scale=%s",
                self.calibration_freeze_backbone,
                self.calibration_head_lr_scale,
            )

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(
            not (self.use_calibration_head and self.calibration_freeze_backbone)
        )

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        if self.use_calibration_head:
            self.optimizer = Adam(
                [
                    {"params": self.model.unet.parameters(), "lr": lr},
                    {
                        "params": self.calibration_head.parameters(),
                        "lr": lr * self.calibration_head_lr_scale,
                    },
                ]
            )
        else:
            self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics (clip_compare_* use compare_single_frame_valid-style meter bounds)
        clip_min = cfg.eval.get("compare_clip_min_m", 2.0)
        clip_max = cfg.eval.get("compare_clip_max_m", 80.0)
        ignore_top_rows = cfg.eval.get("compare_ignore_top_rows", 0)
        self.metric_names, self.metric_funcs = metric.build_eval_metric_callables(
            cfg.eval.eval_metrics,
            compare_clip_min_m=clip_min,
            compare_clip_max_m=clip_max,
            compare_ignore_top_rows=ignore_top_rows,
        )
        self.vis_depth_min_m = cfg.eval.get("vis_magma_min_m", clip_min)
        self.vis_depth_max_m = cfg.eval.get("vis_magma_max_m", clip_max)
        _train_metric_keys = ["loss"]
        if self.use_calibration_head:
            _train_metric_keys.extend(
                [
                    "calibration_loss",
                    "calibration_st_loss",
                    "calibration_metric_loss",
                    "calibration_scale_mean",
                    "calibration_shift_mean",
                    "calibration_scale_target_mean",
                    "calibration_shift_target_mean",
                    "calibration_weight",
                ]
            )
        self.train_metrics = MetricTracker(*_train_metric_keys)
        self.val_metrics = MetricTracker(*self.metric_names)
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.model.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return

    def _get_denoised_latents(self, model_pred, target, noisy_latents, timesteps):
        alphas_cumprod = self.training_noise_scheduler.alphas_cumprod.to(
            device=noisy_latents.device, dtype=noisy_latents.dtype
        )
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        if "sample" == self.prediction_type:
            denoised_pred = model_pred
            denoised_gt = target
        elif "epsilon" == self.prediction_type:
            denoised_pred = (
                noisy_latents - sqrt_one_minus_alpha_t * model_pred
            ) / sqrt_alpha_t.clamp_min(1e-12)
            denoised_gt = (
                noisy_latents - sqrt_one_minus_alpha_t * target
            ) / sqrt_alpha_t.clamp_min(1e-12)
        elif "v_prediction" == self.prediction_type:
            denoised_pred = (
                sqrt_alpha_t * noisy_latents - sqrt_one_minus_alpha_t * model_pred
            )
            denoised_gt = (
                sqrt_alpha_t * noisy_latents - sqrt_one_minus_alpha_t * target
            )
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        return denoised_pred, denoised_gt

    def _normalized_to_metric_depth(self, depth_norm):
        depth_norm_type = OmegaConf.select(
            self.cfg, "depth_normalization.type", default="reversible_normalization"
        )
        if depth_norm_type == "log_inverse_normalization":
            norm_min = float(self.cfg.depth_normalization.norm_min)
            norm_max = float(self.cfg.depth_normalization.norm_max)
            max_depth = float(self.cfg.depth_normalization.max_depth)
            depth_unit = (depth_norm - norm_min) / max(norm_max - norm_min, 1e-8)
            log_min = torch.log(
                torch.tensor(1.0 / (max_depth + 1e-6), device=depth_norm.device)
            )
            log_max = torch.log(torch.tensor(1.0 / 1e-6, device=depth_norm.device))
            log_disp = depth_unit * (log_max - log_min) + log_min
            depth_metric = 1.0 / torch.exp(log_disp)
        else:
            depth_metric = self.depth_transform.denormalize(depth_norm)
        return depth_metric.clamp_min(1e-6)

    def _calibration_weight(self) -> float:
        if not self.use_calibration_head:
            return 0.0
        if self.effective_iter < self.calibration_warmup_iters:
            return 0.0
        if self.calibration_ramp_iters <= 0:
            return 1.0
        progress = (self.effective_iter - self.calibration_warmup_iters) / float(
            self.calibration_ramp_iters
        )
        return float(np.clip(progress, 0.0, 1.0))

    def _pred_to_metric_for_eval(
        self, depth_pred: np.ndarray, depth_raw: np.ndarray, valid_mask: np.ndarray
    ):
        norm_min = float(self.cfg.depth_normalization.norm_min)
        norm_max = float(self.cfg.depth_normalization.norm_max)
        depth_pred_norm = depth_pred * (norm_max - norm_min) + norm_min
        depth_pred_norm_ts = torch.from_numpy(depth_pred_norm).to(dtype=torch.float32)
        depth_metric_base = (
            self.depth_transform.denormalize(depth_pred_norm_ts).cpu().numpy()
        )
        depth_metric_base = np.clip(depth_metric_base, a_min=1e-6, a_max=None)

        if "absolute" == self.cfg.eval.alignment:
            depth_metric = depth_metric_base
        elif "inverse_log" == self.cfg.eval.alignment:
            depth_metric = depth_metric_base
        elif "least_square" == self.cfg.eval.alignment:
            depth_metric, _, _ = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_metric_base,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=self.cfg.eval.align_max_res,
            )
        elif "quantile" == self.cfg.eval.alignment:
            depth_metric, _, _ = align_depth_quantile_map(
                gt_arr=depth_raw,
                pred_arr=depth_metric_base,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=self.cfg.eval.align_max_res,
            )
        else:
            raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")
        return depth_metric

    def _apply_calibration_head_metric(self, depth_metric: np.ndarray) -> np.ndarray:
        if not self.use_calibration_head:
            return depth_metric
        depth_ts = (
            torch.from_numpy(depth_metric)
            .to(self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scale, shift = self.calibration_head(depth_ts)
        depth_cal = apply_affine(depth_ts, scale, shift)
        return depth_cal.squeeze(0).squeeze(0).detach().cpu().numpy()

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()
                if self.use_calibration_head:
                    self.calibration_head.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rgb = batch["rgb_norm"].to(device)
                depth_gt_for_latent = batch[self.gt_depth_type].to(device)

                if self.gt_mask_type is not None:
                    valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
                    valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                else:
                    raise NotImplementedError

                batch_size = rgb.shape[0]

                with torch.no_grad():
                    # Encode image
                    rgb_latent = self.model.encode_rgb(rgb)  # [B, 4, h, w]
                    # Encode GT depth
                    gt_depth_latent = self.encode_depth(
                        depth_gt_for_latent
                    )  # [B, 4, h, w]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        gt_depth_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        gt_depth_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    gt_depth_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Concat rgb and depth latents
                cat_latents = torch.cat(
                    [rgb_latent, noisy_latents], dim=1
                )  # [B, 8, h, w]
                cat_latents = cat_latents.float()

                # Predict the noise residual
                model_pred = self.model.unet(
                    cat_latents, timesteps, text_embed
                ).sample  # [B, 4, h, w]
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                # Get the target for loss depending on the prediction type
                if "sample" == self.prediction_type:
                    target = gt_depth_latent
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        gt_depth_latent, noise, timesteps
                    )  # [B, 4, h, w]
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")
                
                valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                

                # Masked latent loss
                if self.gt_mask_type is not None:
                    if (self.cfg.loss.name == "mse_loss_with_ssim" or self.cfg.loss.name == "kl_divergence"):
                        latent_loss = self.loss(
                            model_pred,
                            target,
                            valid_mask_down,
                        )
                    else:
                        latent_loss = self.loss(
                            model_pred[valid_mask_down].float(),
                            target[valid_mask_down].float(),
                        )
                else:
                    latent_loss = self.loss(model_pred.float(), target.float())

                loss = latent_loss.mean()

                calibration_loss_value = None
                calibration_st_loss_value = None
                calibration_metric_loss_value = None
                calibration_scale_pred = None
                calibration_shift_pred = None
                calibration_scale_target = None
                calibration_shift_target = None
                calibration_weight = 0.0

                if self.use_calibration_head:
                    denoised_pred, _ = self._get_denoised_latents(
                        model_pred=model_pred,
                        target=target,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                    )
                    depth_pred_norm_for_cal = self.model.decode_depth(denoised_pred)
                    depth_pred_metric_for_cal = self._normalized_to_metric_depth(
                        depth_pred_norm_for_cal
                    )
                    depth_gt_metric_for_cal = self._normalized_to_metric_depth(
                        depth_gt_for_latent
                    )

                    with torch.no_grad():
                        (
                            calibration_scale_target,
                            calibration_shift_target,
                        ) = solve_affine_least_squares_batch(
                            pred_depth_metric=depth_pred_metric_for_cal.detach(),
                            gt_depth_metric=depth_gt_metric_for_cal.detach(),
                            valid_mask=valid_mask_for_latent,
                            eps=1e-6,
                        )

                    head_input = (
                        depth_pred_metric_for_cal.detach()
                        if self.calibration_freeze_backbone
                        else depth_pred_metric_for_cal
                    )
                    calibration_scale_pred, calibration_shift_pred = (
                        self.calibration_head(head_input)
                    )
                    depth_calibrated = apply_affine(
                        depth_pred_metric_for_cal,
                        calibration_scale_pred,
                        calibration_shift_pred,
                    )

                    calibration_st_loss_value = F.smooth_l1_loss(
                        calibration_scale_pred, calibration_scale_target
                    ) + self.calibration_lambda_t * F.smooth_l1_loss(
                        calibration_shift_pred, calibration_shift_target
                    )
                    calibration_metric_loss_value = masked_l1_mean(
                        depth_calibrated,
                        depth_gt_metric_for_cal,
                        valid_mask_for_latent,
                    )
                    calibration_loss_value = (
                        self.calibration_lambda_st * calibration_st_loss_value
                        + self.calibration_lambda_metric
                        * calibration_metric_loss_value
                    )
                    calibration_weight = self._calibration_weight()
                    loss = loss + calibration_weight * calibration_loss_value

                ################ Edge loss ########################

                edge_loss_alpha = self.cfg.loss.edge_loss_alpha
                if (edge_loss_alpha > 0):
                    #### Edge loss on the predicted depth ####

                    # # Call the step function with reshaped timesteps
                    # depth_latent_x0 = self.training_noise_scheduler.step(
                    #     model_pred, timestepsV, noisy_latents
                    # ).pred_original_sample                

                    # Reshape timesteps to match the batch dimension of model_pred and noisy_latents
                    timestepsV = timesteps.view(-1, 1, 1, 1)  # Shape [16, 1, 1, 1]

                    # Initialize a list to store predictions for each timestep
                    depth_latent_x0_all = []
                    depth_pred_all = []

                    # Loop over each value in timesteps
                    for i in range(timesteps.shape[0]):
                        single_timestep = timesteps[i].item()  # Get the scalar value for the current batch element
                        
                        # Call step with scalar timestep
                        depth_latent_x0_i = self.training_noise_scheduler.step(
                            model_pred[i:i+1], single_timestep, noisy_latents[i:i+1]
                        ).pred_original_sample
                        
                        # Append the result to the list
                        # depth_latent_x0_all.append(depth_latent_x0_i)
                        depth_pred_i = self.model.decode_depth(depth_latent_x0_i)
                        depth_pred_all.append(depth_pred_i)
                        del depth_pred_i
                        del depth_latent_x0_i
                        torch.cuda.empty_cache()

                    # Concatenate the results to restore batch dimension
                    # depth_latent_x0 = torch.cat(depth_latent_x0_all, dim=0)
                    depth_pred = torch.cat(depth_pred_all, dim=0)
                    del depth_pred_all

                    # Decode the predicted depth
                    depth_pred_edge = get_edge_map(depth_pred)
                    depth_gt_edge = get_edge_map(depth_gt_for_latent)

                    # import ipdb; ipdb.set_trace()
                    # for i in range(timesteps.shape[0]):
                    #     single_timestep = timesteps[i].item()
                    #     d1 = depth_pred[i,:].squeeze().detach().cpu().numpy()
                    #     d1 = np.uint8(255*((d1 - d1.min()) / (d1.max() - d1.min())))

                    #     d2 = depth_gt_for_latent[i,:].squeeze().detach().cpu().numpy()
                    #     d2 = np.uint8(255*((d2 - d2.min()) / (d2.max() - d2.min())))

                    #     r1 = rgb[i,:].squeeze().permute(1,2,0).detach().cpu().numpy()
                    #     r1 = np.uint8(255*((r1 - r1.min()) / (r1.max() - r1.min())))

                    #     renc2dec = self.model.decode_rgb(rgb_latent[i,:].unsqueeze(0))
                    #     r2 = renc2dec.squeeze().permute(1,2,0).detach().cpu().numpy()
                    #     r2 = np.uint8(255*((r2 - r2.min()) / (r2.max() - r2.min())))

                    #     e1 = depth_pred_edge[i,:].squeeze().detach().cpu().numpy()
                    #     e1 = np.uint8(255*((e1 - e1.min()) / (e1.max() - e1.min())))

                    #     e2 = depth_gt_edge[i,:].squeeze().detach().cpu().numpy()
                    #     e2 = np.uint8(255*((e2 - e2.min()) / (e2.max() - e2.min())))

                    #     os.makedirs("debugging/depth_images/", exist_ok=True)
                    #     io.imwrite(f"debugging/depth_images/{single_timestep}_depth_pred.png", d1)
                    #     io.imwrite(f"debugging/depth_images/{single_timestep}_depth_gt.png", d2)
                    #     io.imwrite(f"debugging/depth_images/{single_timestep}_rgb.png", r1)
                    #     io.imwrite(f"debugging/depth_images/{single_timestep}_rgb_reconstructed.png", r2)
                    #     io.imwrite(f"debugging/depth_images/{single_timestep}_depth_pred_edge.png", e1)
                    #     io.imwrite(f"debugging/depth_images/{single_timestep}_depth_gt_edge.png", e2)
                    ###

                    # Masked latent loss
                    if self.gt_mask_type is not None:
                        if (self.cfg.loss.name == "mse_loss_with_ssim" or self.cfg.loss.name == "kl_divergence"):
                            edge_loss = self.loss(
                                depth_pred_edge,
                                depth_gt_edge,
                                valid_mask_down,
                            )
                        else:
                            edge_loss = self.loss(
                                depth_pred_edge[valid_mask_for_latent].float(),
                                depth_gt_edge[valid_mask_for_latent].float(),
                            )
                    else:
                        edge_loss = self.loss(depth_pred_edge.float(), depth_gt_edge.float())
                    
                    loss += edge_loss_alpha * edge_loss.mean()

                ################## Edge loss ######################


                self.train_metrics.update("loss", loss.item())
                if self.use_calibration_head and calibration_loss_value is not None:
                    self.train_metrics.update(
                        "calibration_loss", calibration_loss_value.item()
                    )
                    self.train_metrics.update(
                        "calibration_st_loss", calibration_st_loss_value.item()
                    )
                    self.train_metrics.update(
                        "calibration_metric_loss",
                        calibration_metric_loss_value.item(),
                    )
                    self.train_metrics.update(
                        "calibration_scale_mean",
                        calibration_scale_pred.mean().item(),
                    )
                    self.train_metrics.update(
                        "calibration_shift_mean",
                        calibration_shift_pred.mean().item(),
                    )
                    self.train_metrics.update(
                        "calibration_scale_target_mean",
                        calibration_scale_target.mean().item(),
                    )
                    self.train_metrics.update(
                        "calibration_shift_target_mean",
                        calibration_shift_target.mean().item(),
                    )
                    self.train_metrics.update("calibration_weight", calibration_weight)

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dic(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.debug(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self, save_best_checkpoint=True):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            self._append_eval_metrics_csv(
                dataset_name=val_dataset_name, val_metric_dic=val_metric_dic
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    if save_best_checkpoint:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                        )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        self.model.unet.eval()
        self.model.vae.eval()
        self.model.text_encoder.eval()
        if self.use_calibration_head:
            self.calibration_head.eval()
        metric_tracker.reset()
        per_sample_metrics = {k: [] for k in self.metric_names}
        wandb_vis_images = []

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_int"]  # [B, 3, H, W]
            # GT depth
            depth_raw_ts = batch["depth_raw_linear"].squeeze()
            depth_raw = depth_raw_ts.numpy()
            depth_raw_ts = depth_raw_ts.to(self.device)
            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            valid_mask = valid_mask_ts.numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out: MarigoldDepthOutput = self.model(
                rgb_int,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map="magma",
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            depth_pred = self._pred_to_metric_for_eval(
                depth_pred=depth_pred,
                depth_raw=depth_raw,
                valid_mask=valid_mask,
            )
            if (
                self.use_calibration_head
                and self.calibration_apply_in_validation
                and self.cfg.eval.alignment not in ("least_square", "quantile")
            ):
                depth_pred = self._apply_calibration_head_metric(depth_pred)

            # Clip to dataset min max
            depth_pred = np.clip(
                depth_pred,
                a_min=data_loader.dataset.min_depth,
                a_max=data_loader.dataset.max_depth,
            )

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for _metric_name, met_func in zip(self.metric_names, self.metric_funcs):
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)
                per_sample_metrics[_metric_name].append(_metric)

            # Save SVG visualizations. Per-panel folders intentionally omit colorbars;
            # vis_all keeps the colorbar overlay for quick inspection.
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                svg_name = f"{os.path.splitext(img_name)[0]}.svg"

                pseudo_depth = None
                if "pseudo_depth_linear" in batch:
                    pseudo_depth = batch["pseudo_depth_linear"].squeeze().numpy()

                rgb_img = np.moveaxis(rgb_int.detach().cpu().numpy().squeeze(), 0, -1)
                panels = get_visualization_panels(
                    rgb_img,
                    depth_raw,
                    depth_pred,
                    pseudo_depth=pseudo_depth,
                    valid_mask=valid_mask,
                    depth_vis_min_m=self.vis_depth_min_m,
                    depth_vis_max_m=self.vis_depth_max_m,
                )
                panel_dirs = {
                    "event": "event",
                    "ground_truth": "ground_truth",
                    "prediction_valid": "prediction_valid",
                    "prediction": "prediction",
                }
                for panel_key, folder_name in panel_dirs.items():
                    panel_dir = os.path.join(save_to_dir, folder_name)
                    os.makedirs(panel_dir, exist_ok=True)
                    save_rgb_svg(os.path.join(panel_dir, svg_name), panels[panel_key])

                colorbar_dir = os.path.join(save_to_dir, "colorbar")
                save_magma_colorbar_svg(
                    os.path.join(colorbar_dir, "magma_colorbar.svg"),
                    depth_raw.shape[0],
                    depth_vis_min_m=self.vis_depth_min_m,
                    depth_vis_max_m=self.vis_depth_max_m,
                )

                img_vis = get_visualization_image(
                    rgb_img,
                    depth_raw,
                    depth_pred,
                    pseudo_depth=pseudo_depth,
                    valid_mask=valid_mask,
                    depth_vis_min_m=self.vis_depth_min_m,
                    depth_vis_max_m=self.vis_depth_max_m,
                )
                vis_all_path = os.path.join(save_to_dir, f"vis_all_{svg_name}")
                save_rgb_svg(vis_all_path, img_vis)
                wandb_vis_images.append(wandb.Image(img_vis, caption=img_name))


        if save_to_dir is not None and len(wandb_vis_images) > 0:
            try:
                wandb.log(
                    {f"visualization/{data_loader.dataset.disp_name}": wandb_vis_images},
                    step=self.effective_iter,
                )
            except Exception as exc:
                logging.warning("Failed to log validation images to W&B: %s", exc)

        mean_metrics = metric_tracker.result()
        if bool(OmegaConf.select(self.cfg, "eval.log_std", default=True)):
            for k, vals in per_sample_metrics.items():
                if len(vals) == 0:
                    mean_metrics[f"{k}_std"] = float("nan")
                else:
                    mean_metrics[f"{k}_std"] = float(
                        np.nanstd(np.asarray(vals, dtype=np.float64))
                    )
        return mean_metrics

    def _append_eval_metrics_csv(self, dataset_name: str, val_metric_dic: dict):
        csv_dir = os.path.join(self.out_dir_eval, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{dataset_name}.csv")
        row = {
            "iter": int(self.effective_iter),
            "dataset": dataset_name,
            "alignment": str(self.cfg.eval.alignment),
            "compare_clip_min_m": float(self.cfg.eval.get("compare_clip_min_m", 0.0)),
            "compare_clip_max_m": float(self.cfg.eval.get("compare_clip_max_m", 80.0)),
            "compare_ignore_top_rows": int(self.cfg.eval.get("compare_ignore_top_rows", 0)),
        }
        row.update({k: float(v) for k, v in val_metric_dic.items()})
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        os.makedirs(ckpt_dir, exist_ok=True)

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if self.use_calibration_head:
            calibration_head_path = os.path.join(ckpt_dir, "calibration_head.ckpt")
            torch.save(self.calibration_head.state_dict(), calibration_head_path)
            logging.info(f"Calibration head is saved to: {calibration_head_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            if self.use_calibration_head:
                state["calibration_head"] = self.calibration_head.state_dict()
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True,
        require_calibration_head=False,
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        calibration_head_path = os.path.join(ckpt_path, "calibration_head.ckpt")
        trainer_state_path = os.path.join(ckpt_path, "trainer.ckpt")
        checkpoint = None
        calibration_loaded = False
        if (
            not self.use_calibration_head
            and os.path.exists(calibration_head_path)
        ):
            logging.warning(
                "Checkpoint contains calibration_head.ckpt, but the current config "
                "has loss.calibration.enabled=false; calibration weights will not be "
                "loaded or applied. Use the original run config or enable "
                "loss.calibration.enabled for init/eval."
            )
        if self.use_calibration_head and os.path.exists(calibration_head_path):
            self.calibration_head.load_state_dict(
                torch.load(calibration_head_path, map_location=self.device)
            )
            calibration_loaded = True
            logging.info(f"Calibration head is loaded from {calibration_head_path}")
        elif self.use_calibration_head:
            if os.path.exists(trainer_state_path):
                checkpoint = torch.load(trainer_state_path, map_location=self.device)
                calibration_state = checkpoint.get("calibration_head")
                if calibration_state is not None:
                    self.calibration_head.load_state_dict(calibration_state)
                    calibration_loaded = True
                    logging.info(
                        "Calibration head is loaded from trainer state: %s",
                        trainer_state_path,
                    )
                else:
                    logging.warning(
                        "Calibration head is enabled but neither %s nor "
                        "`calibration_head` in %s was found.",
                        calibration_head_path,
                        trainer_state_path,
                    )
            else:
                logging.warning(
                    "Calibration head is enabled but %s was not found.",
                    calibration_head_path,
                )

        if require_calibration_head and self.use_calibration_head and not calibration_loaded:
            raise FileNotFoundError(
                "Calibrated evaluation requires a trained calibration head in "
                f"{calibration_head_path} or {trainer_state_path}. "
                "To evaluate a Stage-I model without a head, use a Stage-I evaluation config."
            )

        # Load training states
        if load_trainer_state:
            if checkpoint is None:
                checkpoint = torch.load(trainer_state_path, map_location=self.device)
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            if self.use_calibration_head and "calibration_head" in checkpoint:
                self.calibration_head.load_state_dict(checkpoint["calibration_head"])
                logging.info(f"calibration head state is loaded from {ckpt_path}")

            optimizer_state = checkpoint.get("optimizer")
            optimizer_loaded = False
            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                    optimizer_loaded = True
                    logging.info(f"optimizer state is loaded from {ckpt_path}")
                except ValueError as exc:
                    logging.warning(
                        "Skipping optimizer state restore from %s because it is incompatible with the current optimizer: %s",
                        ckpt_path,
                        exc,
                    )

            if resume_lr_scheduler and optimizer_loaded:
                lr_scheduler_state = checkpoint.get("lr_scheduler")
                if lr_scheduler_state is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler_state)
                    logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
