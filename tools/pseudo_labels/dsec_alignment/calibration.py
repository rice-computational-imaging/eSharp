# Adapted from the collaborative Diffused-Remap project; see PSEUDO_LABELS.md.
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def _intrinsics(values: list[float]) -> np.ndarray:
    fx, fy, cx, cy = values
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _homogeneous_rotation(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


@dataclass(frozen=True)
class DSECCalibration:
    """Calibration between rectified left frame camera 1 and event camera 0."""

    K_event: np.ndarray
    K_frame: np.ndarray
    frame_to_event: np.ndarray
    event_to_frame: np.ndarray
    event_size: tuple[int, int]
    frame_size: tuple[int, int]
    event_stereo_baseline_m: float


def load_dsec_calibration(path: str | Path) -> DSECCalibration:
    """Load DSEC cam_to_cam.yaml using the documented T_cn_cnm1 convention."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    intr = data["intrinsics"]
    ext = data["extrinsics"]

    r0 = _homogeneous_rotation(np.asarray(ext["R_rect0"], dtype=float))
    r1 = _homogeneous_rotation(np.asarray(ext["R_rect1"], dtype=float))
    t10 = np.asarray(ext["T_10"], dtype=float)  # raw cam0 -> raw cam1
    event_to_frame = r1 @ t10 @ np.linalg.inv(r0)
    frame_to_event = np.linalg.inv(event_to_frame)

    # Left event cam0 -> right event cam3; used by DSEC disparity depth.
    t30 = np.asarray(ext["T_32"], dtype=float) @ np.asarray(ext["T_21"], dtype=float) @ t10
    baseline = float(np.linalg.norm(t30[:3, 3]))
    ew, eh = intr["camRect0"]["resolution"]
    fw, fh = intr["camRect1"]["resolution"]
    return DSECCalibration(
        K_event=_intrinsics(intr["camRect0"]["camera_matrix"]),
        K_frame=_intrinsics(intr["camRect1"]["camera_matrix"]),
        frame_to_event=frame_to_event,
        event_to_frame=event_to_frame,
        event_size=(int(ew), int(eh)),
        frame_size=(int(fw), int(fh)),
        event_stereo_baseline_m=baseline,
    )
