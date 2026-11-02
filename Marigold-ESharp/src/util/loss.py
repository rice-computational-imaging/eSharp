# Author: Bingxin Ke
# Last modified: 2024-02-22

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


def sliced_wasserstein_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    num_projections: int = 64,
    rescaled: bool = False,
    c: float = 500.0,
) -> torch.Tensor:
    """
    Approximate sliced Wasserstein distance between two point clouds.
    Inputs are expected to be [batch, features] tensors.
    """
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p={p.shape}, q={q.shape}")
    if p.ndim != 2:
        raise ValueError(f"Expected 2D tensors [B, D], got ndim={p.ndim}")

    if rescaled:
        p = p * c
        q = q * c

    dim = p.shape[1]
    dist_acc = 0.0
    for _ in range(num_projections):
        theta = torch.randn(dim, device=p.device, dtype=p.dtype)
        theta = theta / theta.norm(p=2).clamp_min(1e-12)

        proj_p = torch.matmul(p, theta)
        proj_q = torch.matmul(q, theta)
        proj_p, _ = torch.sort(proj_p)
        proj_q, _ = torch.sort(proj_q)
        dist_acc = dist_acc + torch.mean(torch.abs(proj_p - proj_q))

    return dist_acc / num_projections


def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif "mse_loss_with_ssim" == loss_name:
        criterion = MSELossWithSSIM(**kwargs)
    elif "kl_divergence" == loss_name:
        criterion = KLDivergence(**kwargs)
    elif "log_inv_mse" == loss_name:
        criterion = LogInvMSELoss(**kwargs)
    else:
        raise NotImplementedError

    return criterion


class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss
    
class KLDivergence:
    def __init__(self, kl_weight=0.1, reduction="mean") -> None:
        # super().__init__()
        self.kl_weight = kl_weight
        self.reduction = reduction
        pass

    def __call__(self, pred, gt, valid_mask):
        kl_loss = -1 * torch.nn.functional.cosine_similarity(pred, gt)

        pred_flat = pred[valid_mask].float()
        gt_flat = gt[valid_mask].float()
        mse = torch.nn.MSELoss(reduction=self.reduction)
        mse_loss = mse(pred_flat, gt_flat)

        loss = self.kl_weight * kl_loss + mse_loss

        return loss

class MSELossWithSSIM:
    def __init__(self, ssim_weight=0.85, reduction="mean") -> None:
        # super().__init__()
        self.ssim_weight = ssim_weight
        self.reduction = reduction
        pass

    def __call__(self, pred, gt, valid_mask, pred_decoded, gt_decoded):

        pred_flat = pred[valid_mask].float()
        gt_flat = gt[valid_mask].float()
        mse = torch.nn.MSELoss(reduction=self.reduction)
        mse_loss = mse(pred_flat, gt_flat)

        ssim = StructuralSimilarityIndexMeasure().to(pred.device)
        # TODO: apply on decoded depths instead of in latent space
        ssim_loss = 1 - ssim(pred_decoded, gt_decoded)

        loss = self.ssim_weight * ssim_loss + (1 - self.ssim_weight) * mse_loss

        return loss

class LogInvMSELoss:
    def __init__(self, log_inv_pred=False, reduction="mean"):
        # True : if you are predicting log inverse of depth
        self.log_inv_pred = log_inv_pred
        self.reduction = reduction

    def __call__(self, pred, gt):
        if False == self.log_inv_pred:
            inv_log_pred = torch.log(1/(pred+1e-6))
        else:
            inv_log_pred = pred
        inv_log_gt = torch.log(1/(gt+1e-6))
        mse = torch.nn.MSELoss(reduction=self.reduction)
        mse_loss = mse(inv_log_pred, inv_log_gt)

        return mse_loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss
