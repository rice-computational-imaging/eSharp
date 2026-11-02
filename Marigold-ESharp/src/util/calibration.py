import torch
import torch.nn as nn


class DepthCalibrationHead(nn.Module):
    """Predict per-image affine calibration parameters (s, t)."""

    def __init__(self, hidden_dim: int = 32, t_max_m: float = 20.0):
        super().__init__()
        self.t_max_m = float(t_max_m)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, depth_metric: torch.Tensor, valid_mask: torch.Tensor | None = None):
        feat = self.encoder(depth_metric).flatten(1)
        raw = self.mlp(feat)
        alpha = raw[:, 0]
        beta = raw[:, 1]
        scale = torch.exp(alpha).clamp_min(1e-6)
        shift = self.t_max_m * torch.tanh(beta)
        return scale, shift


def build_calibration_head(hidden_dim: int = 32, t_max_m: float = 20.0) -> nn.Module:
    return DepthCalibrationHead(hidden_dim=hidden_dim, t_max_m=t_max_m)


def solve_affine_least_squares_batch(
    pred_depth_metric: torch.Tensor,
    gt_depth_metric: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
):
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    mask = valid_mask.to(dtype=pred_depth_metric.dtype)

    x = pred_depth_metric
    y = gt_depth_metric

    n = mask.sum(dim=(1, 2, 3))
    n_safe = n.clamp_min(1.0)

    x_mean = (mask * x).sum(dim=(1, 2, 3)) / n_safe
    y_mean = (mask * y).sum(dim=(1, 2, 3)) / n_safe

    x_centered = x - x_mean[:, None, None, None]
    y_centered = y - y_mean[:, None, None, None]
    var_x = (mask * x_centered.pow(2)).sum(dim=(1, 2, 3)) / n_safe
    cov_xy = (mask * x_centered * y_centered).sum(dim=(1, 2, 3)) / n_safe

    scale = cov_xy / (var_x + eps)
    shift = y_mean - scale * x_mean

    valid_fit = (n > 1.0) & (var_x > eps)
    scale = torch.where(valid_fit, scale, torch.ones_like(scale))
    shift = torch.where(valid_fit, shift, torch.zeros_like(shift))

    return scale, shift


def apply_calibration_batch(
    calibration_head: nn.Module,
    depth_metric: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale, shift = calibration_head(depth_metric, valid_mask)
    return apply_affine(depth_metric, scale, shift)


def masked_l1_mean(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6):
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    mask = valid_mask.to(dtype=pred.dtype)
    abs_err = torch.abs(pred - gt) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    per_img = abs_err.sum(dim=(1, 2, 3)) / denom
    return per_img.mean()


def apply_affine(depth_metric: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    return scale[:, None, None, None] * depth_metric + shift[:, None, None, None]
