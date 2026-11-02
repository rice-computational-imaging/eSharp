#!/usr/bin/env python3
"""Generate matched SAM 2 automatic-mask overlays from two depth visualizations.

This is a qualitative probe only: SAM 2 is applied to rasterized depth renders,
not evaluated against semantic labels. Both inputs use exactly the same canvas
and automatic-mask parameters.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


SAM2_CONFIG = "configs/sam2/sam2_hiera_s.yaml"
DEFAULT_POINTS_PER_SIDE = 32
DEFAULT_PRED_IOU_THRESH = 0.80
DEFAULT_STABILITY_SCORE_THRESH = 0.92
DEFAULT_MIN_MASK_REGION_AREA = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffused-svg", type=Path, required=True)
    parser.add_argument("--depthanyevent-svg", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def rasterize_svg(path: Path, width: int, height: int) -> np.ndarray:
    png = cairosvg.svg2png(url=str(path), output_width=width, output_height=height)
    image = Image.open(__import__("io").BytesIO(png)).convert("RGB")
    return np.asarray(image)


def overlay_masks(image: np.ndarray, masks: list[dict]) -> np.ndarray:
    overlay = image.astype(np.float32) / 255.0
    rng = np.random.default_rng(20260828)
    # Largest regions first keeps smaller object boundaries visible on top.
    for mask in sorted(masks, key=lambda item: item["area"], reverse=True):
        color = rng.uniform(0.15, 0.95, size=3)
        region = mask["segmentation"]
        overlay[region] = 0.55 * overlay[region] + 0.45 * color

        edge = region ^ np.roll(region, 1, axis=0) | region ^ np.roll(region, 1, axis=1)
        overlay[edge] = 0.0
    return np.clip(overlay, 0.0, 1.0)


def save_svg(image: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    with plt.rc_context({"svg.fonttype": "none", "axes.unicode_minus": False}):
        fig.savefig(path, format="svg", dpi=300, transparent=True, pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sam2_model = build_sam2(SAM2_CONFIG, str(args.sam2_checkpoint), device=args.device)
    generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=DEFAULT_POINTS_PER_SIDE,
        pred_iou_thresh=DEFAULT_PRED_IOU_THRESH,
        stability_score_thresh=DEFAULT_STABILITY_SCORE_THRESH,
        crop_n_layers=0,
        min_mask_region_area=DEFAULT_MIN_MASK_REGION_AREA,
    )

    sources = {
        "diffused": args.diffused_svg,
        "depthanyevent": args.depthanyevent_svg,
    }
    summary: dict[str, object] = {
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "sam2_config": SAM2_CONFIG,
        "input_size": [args.height, args.width],
        "automatic_mask_parameters": {
            "points_per_side": DEFAULT_POINTS_PER_SIDE,
            "pred_iou_thresh": DEFAULT_PRED_IOU_THRESH,
            "stability_score_thresh": DEFAULT_STABILITY_SCORE_THRESH,
            "crop_n_layers": 0,
            "min_mask_region_area": DEFAULT_MIN_MASK_REGION_AREA,
        },
        "note": "Qualitative SAM 2 masks on depth visualizations; not semantic-segmentation metrics.",
    }

    for name, source in sources.items():
        shutil.copy2(source, args.output_dir / f"{name}_depth.svg")
        depth_image = rasterize_svg(source, args.width, args.height)
        masks = generator.generate(depth_image)
        Image.fromarray(depth_image).save(args.output_dir / f"{name}_depth.png")
        save_svg(overlay_masks(depth_image, masks), args.output_dir / f"{name}_sam2_masks.svg")
        summary[name] = {"source": str(source), "mask_count": len(masks)}

    (args.output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
