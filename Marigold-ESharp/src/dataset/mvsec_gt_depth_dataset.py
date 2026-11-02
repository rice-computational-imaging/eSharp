import os
import re

import numpy as np

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class MVSECGTDepthDataset(BaseDepthDataset):
    """
    Read explicit MVSEC GT depth paths or map legacy pseudo-label paths to GT.
    """

    _FRAME_RE = re.compile(r"frame_(\d+)_pred_depth\.npy$")

    def __init__(
        self,
        gt_depth_root: str = "/path/to/data/event3d/mvsec/train",
        **kwargs,
    ) -> None:
        self.gt_depth_root = gt_depth_root
        self._available_idx_cache = {}
        super().__init__(
            min_depth=1e-5,
            max_depth=250.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _map_sequence(self, seq_name: str):
        prefix = "mvsec_dataset_day2_"
        if seq_name.startswith(prefix):
            split_name = seq_name[len(prefix) :]
            return f"mvsec_dataset_day2/{split_name}"
        return seq_name

    def _get_available_indices(self, mapped_seq: str):
        if mapped_seq in self._available_idx_cache:
            return self._available_idx_cache[mapped_seq]
        depth_dir = os.path.join(self.gt_depth_root, mapped_seq, "depth", "data")
        if not os.path.isdir(depth_dir):
            self._available_idx_cache[mapped_seq] = []
            return []
        indices = []
        for name in os.listdir(depth_dir):
            m = re.match(r"depth_(\d+)\.npy$", name)
            if m is not None:
                indices.append(int(m.group(1)))
        indices = sorted(set(indices))
        self._available_idx_cache[mapped_seq] = indices
        return indices

    def _resolve_frame_idx(self, mapped_seq: str, frame_idx: int):
        """
        Strict frame-index resolution for GT eval:
        use exact index only; raise on mismatch to avoid silent duplicate sampling.
        """
        avail = self._get_available_indices(mapped_seq)
        if len(avail) == 0:
            return frame_idx
        avail_set = set(avail)
        if frame_idx in avail_set:
            return frame_idx
        raise FileNotFoundError(
            f"Missing MVSEC GT frame index {frame_idx} in sequence `{mapped_seq}`. "
            f"Available index range: [{avail[0]}, {avail[-1]}]."
        )

    def _pseudo_to_gt_depth_path(self, pseudo_path: str):
        seq_name = os.path.basename(os.path.dirname(pseudo_path))
        m = self._FRAME_RE.search(os.path.basename(pseudo_path))
        if m is None:
            raise ValueError(f"Cannot parse frame index from pseudo path: {pseudo_path}")
        frame_idx = int(m.group(1))
        mapped_seq = self._map_sequence(seq_name)
        resolved_idx = self._resolve_frame_idx(mapped_seq, frame_idx)
        depth_name = f"depth_{resolved_idx:010d}.npy"
        depth_path = os.path.join(self.gt_depth_root, mapped_seq, "depth", "data", depth_name)
        return depth_path

    def _read_depth_file(self, rel_path):
        if re.fullmatch(r'depth_\d+\.npy', os.path.basename(rel_path)):
            depth_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.dataset_dir, rel_path)
        else:
            depth_path = self._pseudo_to_gt_depth_path(rel_path)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Missing MVSEC GT depth file: {depth_path}")
        depth = np.load(depth_path).astype(np.float32).squeeze()
        depth[np.isnan(depth)] = 0.0
        depth[np.isinf(depth)] = 0.0

        factor = 8
        if depth.shape[0] % factor != 0 or depth.shape[1] % factor != 0:
            depth = depth[
                : depth.shape[0] // factor * factor, : depth.shape[1] // factor * factor
            ]
        return depth[np.newaxis, :, :]

    def _load_pseudo_depth_data(self, depth_rel_path):
        if re.fullmatch(r'depth_\d+\.npy', os.path.basename(depth_rel_path)):
            return None
        return super()._load_pseudo_depth_data(depth_rel_path)
