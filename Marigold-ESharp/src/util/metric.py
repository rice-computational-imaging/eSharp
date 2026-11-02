# Author: Bingxin Ke
# Last modified: 2024-02-15

from functools import partial
import pandas as pd
import torch

# Metrics that match Event-WassDiff/compare_single_frame_valid.py (clip pred & GT, then mask).
CLIP_COMPARE_METRICS = frozenset(
    {
        "clip_compare_abs_rel",
        "clip_compare_sq_rel",
        "clip_compare_rmse",
        "clip_compare_rmse_log",
        "clip_compare_silog",
        "clip_compare_lpips",
        "clip_compare_delta1_acc",
        "clip_compare_delta2_acc",
        "clip_compare_delta3_acc",
    }
)

_COMPARE_EPS = 1e-6
_DEPTHANYEVENT_EPS = 1e-5
_LPIPS_MODELS = {}


def build_eval_metric_callables(
    eval_metric_names,
    compare_clip_min_m=2.0,
    compare_clip_max_m=80.0,
    compare_ignore_top_rows=0,
):
    """
    Build (name, fn) pairs for validation. Clip-compare metrics get fixed meter bounds
    (same defaults as compare_single_frame_valid.py --eval_min / --eval_max).
    """
    names = list(eval_metric_names)
    funcs = []
    g = globals()
    for name in names:
        fn = g[name]
        if name.startswith("clip_compare_"):
            fn = partial(
                fn,
                eval_min=compare_clip_min_m,
                eval_max=compare_clip_max_m,
                ignore_top_rows=compare_ignore_top_rows,
            )
        funcs.append(fn)
    return names, funcs


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def abs_difference_10m(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_diff = torch.abs(actual_output - actual_target)
    depth_threshold_mask = actual_target < 10
    if valid_mask is not None:
        valid_mask = valid_mask & depth_threshold_mask
    else:
        valid_mask = depth_threshold_mask
    abs_diff[~valid_mask] = 0
    n = valid_mask.sum((-1, -2))
    abs_diff = torch.sum(abs_diff, (-1, -2)) / n
    return abs_diff.mean()


def abs_difference_20m(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_diff = torch.abs(actual_output - actual_target)
    depth_threshold_mask = actual_target < 20
    if valid_mask is not None:
        valid_mask = valid_mask & depth_threshold_mask
    else:
        valid_mask = depth_threshold_mask
    abs_diff[~valid_mask] = 0
    n = valid_mask.sum((-1, -2))
    abs_diff = torch.sum(abs_diff, (-1, -2)) / n
    return abs_diff.mean()


def abs_difference_30m(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_diff = torch.abs(actual_output - actual_target)
    depth_threshold_mask = actual_target < 30
    if valid_mask is not None:
        valid_mask = valid_mask & depth_threshold_mask
    else:
        valid_mask = depth_threshold_mask
    abs_diff[~valid_mask] = 0
    n = valid_mask.sum((-1, -2))
    abs_diff = torch.sum(abs_diff, (-1, -2)) / n
    return abs_diff.mean()


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2)
        / torch.pow(actual_target, 2)
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss


def _clip_compare_prepare(output, target, valid_mask, eval_min, eval_max, ignore_top_rows=0):
    """
    Match DepthAnyEvent add_to_metrics semantics:
    - clip to [eval_min, eval_max]
    - valid := input valid_mask & finite & (target>0,target<max_target) & (pred>0,pred<max_target)
      where max_target is computed per-sample from (finite) target.
    """

    if output.ndim == 2:
        output = output.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)

    if ignore_top_rows > 0:
        valid_mask = valid_mask.clone()
        valid_mask[..., : int(ignore_top_rows), :] = False

    output = torch.clamp(output, min=eval_min, max=eval_max)
    target = torch.clamp(target, min=eval_min, max=eval_max)

    finite_target = torch.isfinite(target)
    finite_output = torch.isfinite(output)
    finite = finite_target & finite_output

    target_for_max = torch.where(finite_target, target, torch.zeros_like(target))
    max_target = target_for_max.amax(dim=(-1, -2), keepdim=True)

    prediction_mask = (output > 0) & (output < max_target)
    depth_mask = (target > 0) & (target < max_target)
    valid = valid_mask & finite & depth_mask & prediction_mask
    return output, target, valid


def _masked_mean_like_depthanyevent(value, valid):
    value = value * valid.float()
    n = valid.sum(dim=(-1, -2))
    per_sample = value.sum(dim=(-1, -2)) / n.clamp(min=1).float()
    per_sample = torch.where(
        n > 0,
        per_sample,
        torch.full_like(per_sample, float("nan")),
    )
    return per_sample.mean()


def _clip_compare_apply_depth_cap(valid, target, depth_cap_m: float):
    return valid & (target < float(depth_cap_m))


def _get_lpips_model(device: torch.device):
    key = str(device)
    if key in _LPIPS_MODELS:
        return _LPIPS_MODELS[key]
    try:
        import lpips  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "LPIPS metric requested but `lpips` is not installed. "
            "Install it with: pip install lpips"
        ) from e
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    _LPIPS_MODELS[key] = model
    return model


def _depth_to_lpips_3ch(depth, valid):
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    scale = depth.amax(dim=(-1, -2), keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    depth01 = depth / scale
    depth11 = depth01 * 2.0 - 1.0
    return depth11.unsqueeze(1).repeat(1, 3, 1, 1)



def clip_compare_abs_rel(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    abs_diff = torch.abs(output - target)
    r = abs_diff / (target + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_sq_rel(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    abs_diff = torch.abs(output - target)
    r = torch.pow(abs_diff, 2) / (torch.pow(target, 2) + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_rmse(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    diff2 = torch.pow(torch.abs(output - target), 2)
    mse = _masked_mean_like_depthanyevent(diff2, valid)
    return torch.sqrt(mse)


def clip_compare_rmse_log(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    log_diff2 = torch.pow(
        torch.log(target + _DEPTHANYEVENT_EPS)
        - torch.log(output + _DEPTHANYEVENT_EPS),
        2,
    )
    mse = _masked_mean_like_depthanyevent(log_diff2, valid)
    return torch.sqrt(mse)


def clip_compare_silog(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    log_diff = torch.log(target + _DEPTHANYEVENT_EPS) - torch.log(
        output + _DEPTHANYEVENT_EPS
    )
    vf = valid.float()
    n = valid.sum(dim=(-1, -2))
    n_safe = n.clamp(min=1).float()
    mean_diff = (log_diff * vf).sum(dim=(-1, -2)) / n_safe
    mean_diff2 = (torch.pow(log_diff, 2) * vf).sum(dim=(-1, -2)) / n_safe
    silog_per_sample = mean_diff2 - torch.pow(mean_diff, 2)
    silog_per_sample = torch.where(
        n > 0,
        silog_per_sample,
        torch.full_like(silog_per_sample, float("nan")),
    )
    return silog_per_sample.mean()


def clip_compare_lpips(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    model = _get_lpips_model(output.device)
    pred_3ch = _depth_to_lpips_3ch(output, valid)
    gt_3ch = _depth_to_lpips_3ch(target, valid)
    with torch.no_grad():
        val = model(pred_3ch, gt_3ch)
    return val.mean()


def clip_compare_delta1_acc(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    ratio = torch.max(
        target / (output + _DEPTHANYEVENT_EPS),
        output / (target + _DEPTHANYEVENT_EPS),
    )
    return _masked_mean_like_depthanyevent((ratio <= 1.25).float(), valid)


def clip_compare_delta2_acc(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    ratio = torch.max(
        target / (output + _DEPTHANYEVENT_EPS),
        output / (target + _DEPTHANYEVENT_EPS),
    )
    return _masked_mean_like_depthanyevent((ratio <= 1.25**2).float(), valid)


def clip_compare_delta3_acc(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    ratio = torch.max(
        target / (output + _DEPTHANYEVENT_EPS),
        output / (target + _DEPTHANYEVENT_EPS),
    )
    return _masked_mean_like_depthanyevent((ratio <= 1.25**3).float(), valid)


def clip_compare_abs_rel_10m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 10.0)
    abs_diff = torch.abs(output - target)
    r = abs_diff / (target + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_abs_rel_20m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 20.0)
    abs_diff = torch.abs(output - target)
    r = abs_diff / (target + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_abs_rel_30m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 30.0)
    abs_diff = torch.abs(output - target)
    r = abs_diff / (target + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_sq_rel_10m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 10.0)
    abs_diff = torch.abs(output - target)
    r = torch.pow(abs_diff, 2) / (torch.pow(target, 2) + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_sq_rel_20m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 20.0)
    abs_diff = torch.abs(output - target)
    r = torch.pow(abs_diff, 2) / (torch.pow(target, 2) + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_sq_rel_30m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 30.0)
    abs_diff = torch.abs(output - target)
    r = torch.pow(abs_diff, 2) / (torch.pow(target, 2) + _DEPTHANYEVENT_EPS)
    return _masked_mean_like_depthanyevent(r, valid)


def clip_compare_rmse_10m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 10.0)
    diff2 = torch.pow(torch.abs(output - target), 2)
    mse = _masked_mean_like_depthanyevent(diff2, valid)
    return torch.sqrt(mse)


def clip_compare_rmse_20m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 20.0)
    diff2 = torch.pow(torch.abs(output - target), 2)
    mse = _masked_mean_like_depthanyevent(diff2, valid)
    return torch.sqrt(mse)


def clip_compare_rmse_30m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 30.0)
    diff2 = torch.pow(torch.abs(output - target), 2)
    mse = _masked_mean_like_depthanyevent(diff2, valid)
    return torch.sqrt(mse)


def clip_compare_delta1_acc_10m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 10.0)
    ratio = torch.max(
        target / (output + _DEPTHANYEVENT_EPS),
        output / (target + _DEPTHANYEVENT_EPS),
    )
    return _masked_mean_like_depthanyevent((ratio <= 1.25).float(), valid)


def clip_compare_delta1_acc_20m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 20.0)
    ratio = torch.max(
        target / (output + _DEPTHANYEVENT_EPS),
        output / (target + _DEPTHANYEVENT_EPS),
    )
    return _masked_mean_like_depthanyevent((ratio <= 1.25).float(), valid)


def clip_compare_delta1_acc_30m(
    output, target, valid_mask, eval_min=2.0, eval_max=80.0, ignore_top_rows=0
):
    output, target, valid = _clip_compare_prepare(
        output, target, valid_mask, eval_min, eval_max, ignore_top_rows
    )
    valid = _clip_compare_apply_depth_cap(valid, target, 30.0)
    ratio = torch.max(
        target / (output + _DEPTHANYEVENT_EPS),
        output / (target + _DEPTHANYEVENT_EPS),
    )
    return _masked_mean_like_depthanyevent((ratio <= 1.25).float(), valid)
