import os
import re

import cv2
import numpy as np
import yaml

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class DSECEventDisparityDataset(BaseDepthDataset):
    """
    Read explicit DSEC GT disparity PNG paths or legacy pseudo-label paths,
    then convert disparity to metric depth using per-sequence calibration.
    """

    _FRAME_RE = re.compile(r"frame(\d+)_pred_depth\.npy$")

    def __init__(
        self,
        disparity_root: str = "/path/to/data/event3d/DSEC/train/disparity_maps",
        calibration_root: str = "/path/to/data/event3d/DSEC/train/calib",
        q_key: str = "cams_03",
        **kwargs,
    ) -> None:
        self.disparity_root = disparity_root
        self.calibration_root = calibration_root
        self.q_key = q_key
        self._q_cache = {}
        super().__init__(
            min_depth=1e-5,
            max_depth=250.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _load_q(self, seq_name: str):
        if seq_name in self._q_cache:
            return self._q_cache[seq_name]
        candidate_paths = [
            os.path.join(self.calibration_root, seq_name, "cam_to_cam.yaml"),
            os.path.join(
                self.calibration_root, seq_name, "calibration", "cam_to_cam.yaml"
            ),
        ]
        calib_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if calib_path is None:
            tried = "\n".join(candidate_paths)
            raise FileNotFoundError(
                f"Missing calibration file for sequence `{seq_name}`. Tried:\n{tried}"
            )
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f)
        q = np.array(calib["disparity_to_depth"][self.q_key], dtype=np.float32)
        self._q_cache[seq_name] = q
        return q

    def _pseudo_to_disparity_path(self, pseudo_path: str):
        if pseudo_path.lower().endswith('.png'):
            # Explicit GT split rows: <sequence>/disparity/event/<index>.png.
            path = os.path.normpath(pseudo_path)
            event_dir = os.path.dirname(path)
            disparity_dir = os.path.dirname(event_dir)
            if os.path.basename(event_dir) != 'event' or os.path.basename(disparity_dir) != 'disparity':
                raise ValueError(f'Expected <sequence>/disparity/event/<frame>.png: {pseudo_path}')
            seq_name = os.path.basename(os.path.dirname(disparity_dir))
            return seq_name, path if os.path.isabs(path) else os.path.join(self.dataset_dir, path)
        seq_name = os.path.basename(os.path.dirname(pseudo_path))
        m = self._FRAME_RE.search(os.path.basename(pseudo_path))
        if m is None:
            raise ValueError(f"Cannot parse frame index from pseudo path: {pseudo_path}")
        frame_idx = int(m.group(1))
        disp_name = f"{frame_idx:06d}.png"
        disp_path = os.path.join(
            self.disparity_root, seq_name, "disparity", "event", disp_name
        )
        return seq_name, disp_path

    def _read_depth_file(self, rel_path):
        seq_name, disp_path = self._pseudo_to_disparity_path(rel_path)
        if not os.path.exists(disp_path):
            raise FileNotFoundError(f"Missing disparity map: {disp_path}")

        disp_u16 = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        if disp_u16 is None:
            raise RuntimeError(f"Failed to read disparity map: {disp_path}")
        disp = disp_u16.astype(np.float32) / 256.0

        q = self._load_q(seq_name)
        points3d = cv2.reprojectImageTo3D(disp, q)
        depth = points3d[:, :, 2].astype(np.float32)
        depth[disp <= 0] = 0.0
        depth[np.isnan(depth)] = 0.0
        depth[np.isinf(depth)] = 0.0

        factor = 8
        if depth.shape[0] % factor != 0 or depth.shape[1] % factor != 0:
            depth = depth[
                : depth.shape[0] // factor * factor, : depth.shape[1] // factor * factor
            ]
        return depth[np.newaxis, :, :]

    def _load_pseudo_depth_data(self, depth_rel_path):
        # An explicit GT disparity is not a pseudo-depth visualization sidecar.
        if depth_rel_path.lower().endswith('.png'):
            return None
        return super()._load_pseudo_depth_data(depth_rel_path)
