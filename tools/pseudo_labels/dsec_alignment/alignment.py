# Adapted from the collaborative Diffused-Remap project; see PSEUDO_LABELS.md.
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    depth: np.ndarray
    scale: float
    shift: float
    inlier_mask: np.ndarray
    rmse_m: float


def robust_scale_shift(
    relative_depth: np.ndarray,
    metric_depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    max_iterations: int = 10,
    huber_delta: float = 1.5,
    min_points: int = 20,
    min_depth: float = 0.05,
) -> AlignmentResult:
    """Fit metric = scale * relative + shift with IRLS Huber regression."""
    relative = np.asarray(relative_depth, dtype=np.float64)
    metric = np.asarray(metric_depth, dtype=np.float64)
    if relative.shape != metric.shape:
        raise ValueError("relative_depth and metric_depth must have identical shapes")
    valid = np.isfinite(relative) & np.isfinite(metric) & (metric > 0)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if int(valid.sum()) < min_points:
        raise ValueError(f"Need at least {min_points} valid LiDAR samples; got {valid.sum()}")

    x, y = relative[valid], metric[valid]
    design = np.column_stack((x, np.ones_like(x)))
    weights = np.ones_like(y)
    solution = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(max_iterations):
        residual = y - design @ solution
        center = np.median(residual)
        mad = np.median(np.abs(residual - center))
        sigma = max(1.4826 * mad, 1e-6)
        normalized = np.abs(residual - center) / (huber_delta * sigma)
        weights = np.ones_like(normalized)
        np.divide(1.0, normalized, out=weights, where=normalized > 1.0)
        weighted_design = design * np.sqrt(weights[:, None])
        updated = np.linalg.lstsq(weighted_design, y * np.sqrt(weights), rcond=None)[0]
        if np.allclose(updated, solution, rtol=1e-6, atol=1e-8):
            solution = updated
            break
        solution = updated

    scale, shift = map(float, solution)
    if scale <= 0:
        raise ValueError(f"Metric alignment produced non-positive scale {scale:.6g}")
    fitted = np.maximum(scale * relative + shift, min_depth).astype(np.float32)
    residual = y - design @ solution
    sigma = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-6)
    local_inliers = np.abs(residual - np.median(residual)) <= 3.0 * sigma
    inliers = np.zeros(relative.shape, dtype=bool)
    inliers.flat[np.flatnonzero(valid)[local_inliers]] = True
    rmse = float(np.sqrt(np.mean(residual[local_inliers] ** 2)))
    return AlignmentResult(fitted, scale, shift, inliers, rmse)


def robust_inverse_depth_alignment(
    relative_disparity: np.ndarray,
    metric_depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    min_depth: float = 0.05,
) -> AlignmentResult:
    """Align disparity-like output to inverse metric depth, then invert."""
    metric = np.asarray(metric_depth, dtype=np.float64)
    inverse_metric = np.full(metric.shape, np.nan, dtype=np.float64)
    valid_metric = np.isfinite(metric) & (metric > 0)
    inverse_metric[valid_metric] = 1.0 / metric[valid_metric]
    aligned_inverse = robust_scale_shift(
        relative_disparity,
        inverse_metric,
        valid_mask=valid_mask,
        min_depth=np.finfo(np.float32).eps,
    )
    depth = 1.0 / np.maximum(aligned_inverse.depth, np.finfo(np.float32).eps)
    depth = np.maximum(depth, min_depth).astype(np.float32)
    residual = depth[aligned_inverse.inlier_mask] - metric[aligned_inverse.inlier_mask]
    rmse_m = float(np.sqrt(np.mean(residual ** 2)))
    return AlignmentResult(
        depth=depth,
        scale=aligned_inverse.scale,
        shift=aligned_inverse.shift,
        inlier_mask=aligned_inverse.inlier_mask,
        rmse_m=rmse_m,
    )
