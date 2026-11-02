# Author: Bingxin Ke
# Last modified: 2024-01-11

import numpy as np
import torch


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def _downsample_for_alignment(gt, pred, valid_mask, max_resolution):
    if max_resolution is None:
        return gt, pred, valid_mask
    scale_factor = np.min(max_resolution / np.array(gt.shape[-2:]))
    if scale_factor >= 1:
        return gt, pred, valid_mask
    downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
    gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
    pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
    valid_mask = (
        downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float()).bool().numpy()
    )
    return gt, pred, valid_mask


def align_depth_quantile_map(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    """Monotonic quantile mapping from pred to GT (depthanyevent align_quantile_map)."""
    clip_distance = 80.0
    eps = 1e-3
    num_quantiles = 64
    max_depth = np.inf if clip_distance <= 0 else clip_distance

    ori_shape = pred_arr.shape
    gt = gt_arr.squeeze()
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    gt, pred, valid_mask = _downsample_for_alignment(
        gt, pred, valid_mask, max_resolution
    )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    valid = (
        valid_mask
        & np.isfinite(pred)
        & np.isfinite(gt)
        & (pred > eps)
        & (gt > eps)
        & (gt < max_depth)
    )
    if valid.sum() < 32:
        aligned = np.clip(pred.astype(np.float64), 0.0, max_depth)
    else:
        qs = np.linspace(0.0, 1.0, num_quantiles)
        pred_q = np.quantile(pred[valid], qs)
        target_q = np.quantile(gt[valid], qs)
        pred_q, unique_idx = np.unique(pred_q, return_index=True)
        target_q = target_q[unique_idx]
        if pred_q.size < 2:
            aligned = np.clip(pred.astype(np.float64), 0.0, max_depth)
        else:
            aligned = np.interp(
                pred.astype(np.float64),
                pred_q,
                target_q,
                left=target_q[0],
                right=target_q[-1],
            )
            aligned = np.clip(aligned, 0.0, max_depth)

    aligned_pred = aligned.reshape(ori_shape).astype(np.float32)
    if return_scale_shift:
        return aligned_pred, 1.0, 0.0
    return aligned_pred


# ******************** disparity space ********************
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
