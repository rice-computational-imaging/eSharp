# Author: Bingxin Ke
# Last modified: 2024-03-12

import logging
import os
import sys
import wandb
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

def _depth_to_gray_u8(depth_m, h, w, vmin_m, vmax_m):
    """Linear depth (meters) in [vmin, vmax] -> grayscale RGB (same GT vs pred)."""
    d = np.asarray(depth_m, dtype=np.float32)[:h, :w]
    if d.ndim == 3:
        d = d[..., 0]
    d = np.clip(d, vmin_m, vmax_m)
    g = ((d - vmin_m) / max(vmax_m - vmin_m, 1e-6) * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _inverse_log_magma_u8(depth_m, h, w, d_min_m, d_max_m):
    """
    Original Marigold-style inverse-log coloring: clip to [d_min, d_max] (m), then t = log(1/d).

    Per-image min–max on t was removed: that rescales contrast independently each frame, so the
    same meter depth could look different from image to image. Here t is mapped to [0, 1] using
    fixed endpoints at d_max (far) and d_min (near), so colors stay consistent across frames.
    """
    d_min_safe = max(float(d_min_m), 1e-6)
    d_max_safe = max(float(d_max_m), d_min_safe + 1e-6)
    d = np.asarray(depth_m, dtype=np.float32)[:h, :w]
    if d.ndim == 3:
        d = d[..., 0]
    invalid_mask = (~np.isfinite(d)) | (d <= 0)
    d = np.clip(d, d_min_safe, d_max_safe)
    t = np.log(1.0 / np.maximum(d, 1e-6))
    t_far = np.log(1.0 / d_max_safe)
    t_near = np.log(1.0 / d_min_safe)
    u = (t - t_far) / (t_near - t_far + 1e-8)
    u = np.clip(u, 0.0, 1.0)
    rgb = plt.cm.magma(u)[:, :, :3]
    rgb_u8 = (rgb * 255).astype(np.uint8)
    rgb_u8[invalid_mask] = 0
    return rgb_u8


def inverse_log_magma_numpy(
    depth_m: np.ndarray, d_min_m: float = 2.0, d_max_m: float = 100.0
) -> np.ndarray:
    """Full HxW depth (meters) -> HxWx3 uint8 magma (inverse-log + global range)."""
    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape
    return _inverse_log_magma_u8(depth_m, h, w, d_min_m, d_max_m)


DEFAULT_SVG_FONT_FAMILY = "DejaVu Sans"


def editable_svg_rc_params(font_family: str = DEFAULT_SVG_FONT_FAMILY):
    params = {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [font_family],
        "axes.unicode_minus": False,
    }
    if font_family == "DejaVu Sans":
        params["mathtext.fontset"] = "dejavusans"
    return params


def save_editable_svg(
    fig,
    path: str,
    font_family: str = DEFAULT_SVG_FONT_FAMILY,
    bbox_inches: str | None = "tight",
    dpi: int = 300,
    transparent: bool = True,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with plt.rc_context(editable_svg_rc_params(font_family=font_family)):
        fig.savefig(
            path,
            format="svg",
            bbox_inches=bbox_inches,
            dpi=dpi,
            transparent=transparent,
        )
    return path


def save_rgb_svg(path: str, rgb_u8: np.ndarray, dpi: int = 300):
    """Save an RGB uint8 image to an SVG container. Pixel panels remain raster data."""
    img = np.asarray(rgb_u8, dtype=np.uint8)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, interpolation="nearest")
    save_editable_svg(fig, path, bbox_inches=None, dpi=dpi, transparent=True)
    plt.close(fig)


def _magma_colorbar_u8(
    height: int,
    d_min_m: float,
    d_max_m: float,
    width: int = 36,
    meter_ticks=(2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
) -> np.ndarray:
    """
    Render a vertical magma colorbar as uint8 RGB.
    Tick placement matches inverse-log mapping used for depth visualization.
    """
    d_min_safe = max(float(d_min_m), 1e-6)
    d_max_safe = max(float(d_max_m), d_min_safe + 1e-6)

    # Keep only ticks inside display range.
    ticks_m = [float(t) for t in meter_ticks if d_min_safe <= float(t) <= d_max_safe]
    if len(ticks_m) == 0:
        ticks_m = [d_min_safe, d_max_safe]

    denom = np.log(float(d_max_safe) / float(d_min_safe) + 1e-12)
    tick_locs = [np.log(float(d_max_safe) / t) / denom for t in ticks_m]

    fig_h = max(height / 100.0, 2.0)
    fig_w = max(width / 100.0, 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    # origin='lower' so 0 is bottom, 1 is top
    ax.imshow(grad, cmap="magma", origin="lower", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([loc * (height - 1) for loc in tick_locs])
    ax.set_yticklabels([f"{int(t)}" for t in ticks_m], fontsize=8)
    ax.yaxis.tick_right()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.1)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # Matplotlib 3.10+ removed tostring_rgb() on FigureCanvasAgg.
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        img = rgba[:, :, :3]
    else:
        # Backward-compatible fallback for older Matplotlib versions.
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img



def _convert_rgb_to_u8(img, h, w):
    if (img.shape[0] != h) or (img.shape[1] != w):
        img = img[:h, :w]
    if len(img.shape) == 2:
        img = img[:, :, None].repeat(3, axis=-1)

    if img.dtype == np.float32:
        if img.max() > 10:
            img = np.clip(img, a_min=0, a_max=250.0)
            img = img.astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    return img


def get_magma_colorbar_image(
    height: int,
    depth_vis_min_m=2.0,
    depth_vis_max_m=100.0,
    width: int = 36,
):
    return _magma_colorbar_u8(
        height,
        depth_vis_min_m,
        depth_vis_max_m,
        width=width,
    )


def save_magma_colorbar_svg(
    path: str,
    height: int,
    depth_vis_min_m=2.0,
    depth_vis_max_m=100.0,
    width: int = 72,
    meter_ticks=(2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    dpi: int = 300,
):
    """Save a scalable SVG magma colorbar with vector gradient and editable text."""
    del dpi  # Kept for API symmetry with save_editable_svg; this output is vector.
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    d_min_safe = max(float(depth_vis_min_m), 1e-6)
    d_max_safe = max(float(depth_vis_max_m), d_min_safe + 1e-6)
    ticks_m = [float(t) for t in meter_ticks if d_min_safe <= float(t) <= d_max_safe]
    if len(ticks_m) == 0:
        ticks_m = [d_min_safe, d_max_safe]

    h = max(int(height), 32)
    w = max(int(width), 48)
    pad_top = 4
    pad_bottom = 4
    bar_x = 6
    bar_w = max(16, min(28, w // 3))
    bar_h = h - pad_top - pad_bottom
    label_x = bar_x + bar_w + 8
    denom = np.log(float(d_max_safe) / float(d_min_safe) + 1e-12)

    stops = []
    for i, u in enumerate(np.linspace(0.0, 1.0, 64)):
        rgb = (np.asarray(plt.cm.magma(float(u))[:3]) * 255).astype(np.uint8)
        stops.append(
            f'<stop offset="{100.0 * i / 63.0:.4f}%" stop-color="#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" />'
        )

    tick_elems = []
    for tick in ticks_m:
        loc = np.log(float(d_max_safe) / tick) / denom
        y = pad_top + (1.0 - loc) * bar_h
        label = str(int(tick)) if float(tick).is_integer() else f"{tick:g}"
        tick_elems.append(
            f'<line x1="{bar_x + bar_w}" y1="{y:.3f}" x2="{bar_x + bar_w + 5}" y2="{y:.3f}" stroke="black" stroke-width="1" />'
        )
        tick_elems.append(
            f'<text x="{label_x}" y="{y:.3f}" dominant-baseline="middle" '
            f'font-family="DejaVu Sans, sans-serif" font-size="8" fill="black">{label}</text>'
        )

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '  <defs>',
        '    <linearGradient id="magma_depth" x1="0%" y1="100%" x2="0%" y2="0%">',
        *[f'      {stop}' for stop in stops],
        '    </linearGradient>',
        '  </defs>',
        f'  <rect x="{bar_x}" y="{pad_top}" width="{bar_w}" height="{bar_h}" fill="url(#magma_depth)" />',
        f'  <rect x="{bar_x}" y="{pad_top}" width="{bar_w}" height="{bar_h}" fill="none" stroke="black" stroke-width="0.5" />',
        *[f'  {elem}' for elem in tick_elems],
        '</svg>',
        '',
    ]
    with open(path, "w") as f:
        f.write("\n".join(svg_parts))
    return path


def get_visualization_panels(
    image_rgb,
    depth_GT,
    pred_depth,
    pseudo_depth=None,
    valid_mask=None,
    depth_vis_min_m=2.0,
    depth_vis_max_m=100.0,
):
    # Panels match the 5-column vis_all image before colorbars are overlaid.
    h = depth_GT.shape[0]
    w = depth_GT.shape[1]

    event = _convert_rgb_to_u8(image_rgb, h, w)
    ground_truth = _inverse_log_magma_u8(
        depth_GT, h, w, depth_vis_min_m, depth_vis_max_m
    )
    prediction = _inverse_log_magma_u8(
        pred_depth, h, w, depth_vis_min_m, depth_vis_max_m
    )

    pred_depth_valid = np.asarray(pred_depth, dtype=np.float32).copy()
    if valid_mask is None:
        valid_mask_local = np.isfinite(np.asarray(depth_GT)) & (np.asarray(depth_GT) > 0)
    else:
        valid_mask_local = np.asarray(valid_mask).astype(bool)
    if valid_mask_local.ndim == 3:
        valid_mask_local = valid_mask_local[..., 0]
    valid_mask_local = valid_mask_local[:h, :w]
    pred_depth_valid[~valid_mask_local] = 0.0
    prediction_valid = _inverse_log_magma_u8(
        pred_depth_valid, h, w, depth_vis_min_m, depth_vis_max_m
    )
    prediction_valid[~valid_mask_local] = 0

    if pseudo_depth is None:
        pseudo_depth = np.zeros((h, w), dtype=np.float32)
    pseudo_depth = np.asarray(pseudo_depth, dtype=np.float32)
    if pseudo_depth.ndim == 3:
        pseudo_depth = pseudo_depth[..., 0]
    pseudo_canvas = np.zeros((h, w), dtype=np.float32)
    hh = min(h, pseudo_depth.shape[0])
    ww = min(w, pseudo_depth.shape[1])
    pseudo_canvas[:hh, :ww] = pseudo_depth[:hh, :ww]
    pseudo_label = _inverse_log_magma_u8(
        pseudo_canvas, h, w, depth_vis_min_m, depth_vis_max_m
    )

    return {
        "event": event,
        "ground_truth": ground_truth,
        "prediction_valid": prediction_valid,
        "pseudo_label": pseudo_label,
        "prediction": prediction,
    }


def get_visualization_image(
    image_rgb,
    depth_GT,
    pred_depth,
    pseudo_depth=None,
    valid_mask=None,
    depth_vis_min_m=2.0,
    depth_vis_max_m=100.0,
):
    # columns: Event/RGB | GT magma | pred magma (valid) | pseudo-label magma | pred magma (full)
    h = depth_GT.shape[0]
    w = depth_GT.shape[1]
    panels = get_visualization_panels(
        image_rgb,
        depth_GT,
        pred_depth,
        pseudo_depth=pseudo_depth,
        valid_mask=valid_mask,
        depth_vis_min_m=depth_vis_min_m,
        depth_vis_max_m=depth_vis_max_m,
    )

    vis_image = np.zeros((h, 5 * w, 3), dtype=np.uint8)
    vis_image[:, 0:w] = panels["event"]
    vis_image[:, w : 2 * w] = panels["ground_truth"]
    vis_image[:, 2 * w : 3 * w] = panels["prediction_valid"]
    vis_image[:, 3 * w : 4 * w] = panels["pseudo_label"]
    vis_image[:, 4 * w : 5 * w] = panels["prediction"]

    # Add magma colorbars on all magma columns (2nd-5th panels).
    cb = get_magma_colorbar_image(h, depth_vis_min_m, depth_vis_max_m)
    cb_h, cb_w = cb.shape[:2]
    if cb_h != h:
        cb = cb[:h]
    cb_w = min(cb_w, max(8, w // 6))
    cb = cb[:, :cb_w]
    vis_image[:, 2 * w - cb_w : 2 * w] = cb
    vis_image[:, 3 * w - cb_w : 3 * w] = cb
    vis_image[:, 4 * w - cb_w : 4 * w] = cb
    vis_image[:, 5 * w - cb_w : 5 * w] = cb

    return vis_image
    


def config_logging(cfg_logging, out_dir=None):
    file_level = cfg_logging.get("file_level", 10)
    console_level = cfg_logging.get("console_level", 10)

    log_formatter = logging.Formatter(cfg_logging["format"])

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(
            out_dir, cfg_logging.get("filename", "logging.log")
        )
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)


class MyTrainingLogger:
    """Tensorboard + wandb logger"""

    writer: SummaryWriter
    is_initialized = False

    def __init__(self) -> None:
        pass

    def set_dir(self, tb_log_dir):
        if self.is_initialized:
            raise ValueError("Do not initialize writer twice")
        self.writer = SummaryWriter(tb_log_dir)
        self.is_initialized = True

    def log_dic(self, scalar_dic, global_step, walltime=None):
        for k, v in scalar_dic.items():
            self.writer.add_scalar(k, v, global_step=global_step, walltime=walltime)
        return


# global instance
tb_logger = MyTrainingLogger()


# -------------- wandb tools --------------
def init_wandb(enable: bool, **kwargs):
    if enable:
        run = wandb.init(sync_tensorboard=True, **kwargs)
    else:
        run = wandb.init(mode="disabled")
    return run


def log_slurm_job_id(step):
    global tb_logger
    _jobid = os.getenv("SLURM_JOB_ID")
    if _jobid is None:
        _jobid = -1
    tb_logger.writer.add_scalar("job_id", int(_jobid), global_step=step)
    logging.debug(f"Slurm job_id: {_jobid}")


def load_wandb_job_id(out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "r") as f:
        wandb_id = f.read()
    return wandb_id


def save_wandb_job_id(run, out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "w+") as f:
        f.write(run.id)


def eval_dic_to_text(val_metrics: dict, dataset_name: str, sample_list_path: str):
    eval_text = f"Evaluation metrics:\n\
     on dataset: {dataset_name}\n\
     over samples in: {sample_list_path}\n"

    eval_text += tabulate([val_metrics.keys(), val_metrics.values()])
    return eval_text
