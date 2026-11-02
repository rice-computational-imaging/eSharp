"""Affine metric-depth calibration shared by Lotus Stage-II trainers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthCalibrationHead(nn.Module):
    def __init__(self, hidden_dim: int = 32, t_max_m: float = 20.0):
        super().__init__()
        self.t_max_m = float(t_max_m)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(nn.Linear(16, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.mlp(self.encoder(depth).flatten(1))
        return torch.exp(raw[:, 0]).clamp_min(1e-6), self.t_max_m * torch.tanh(raw[:, 1])


def affine_targets(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.to(pred.dtype)
    n = mask.sum((1, 2, 3)).clamp_min(1.0)
    x_mean = (pred * mask).sum((1, 2, 3)) / n
    y_mean = (target * mask).sum((1, 2, 3)) / n
    x_centered = pred - x_mean[:, None, None, None]
    y_centered = target - y_mean[:, None, None, None]
    variance = (mask * x_centered.square()).sum((1, 2, 3)) / n
    covariance = (mask * x_centered * y_centered).sum((1, 2, 3)) / n
    scale = covariance / (variance + 1e-6)
    shift = y_mean - scale * x_mean
    valid = variance > 1e-6
    return torch.where(valid, scale, torch.ones_like(scale)), torch.where(valid, shift, torch.zeros_like(shift))


def masked_l1_mean(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6):
    """Average valid-pixel errors per image, then average over the batch."""
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    mask = valid_mask.to(dtype=pred.dtype)
    abs_err = torch.abs(pred - gt) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    per_img = abs_err.sum(dim=(1, 2, 3)) / denom
    return per_img.mean()


def calibration_loss(head, prediction, target, mask, lambda_st, lambda_t, lambda_metric):
    with torch.no_grad():
        target_scale, target_shift = affine_targets(prediction.detach(), target.detach(), mask)
    scale, shift = head(prediction)
    calibrated = scale[:, None, None, None] * prediction + shift[:, None, None, None]
    st_loss = F.smooth_l1_loss(scale, target_scale) + lambda_t * F.smooth_l1_loss(shift, target_shift)
    metric_loss = masked_l1_mean(calibrated, target, mask)
    return lambda_st * st_loss + lambda_metric * metric_loss, calibrated, st_loss, metric_loss
