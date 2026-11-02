# Last modified: 2024-02-08
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
import numpy as np
import os


class MVSECDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # MVSEC data parameter
            min_depth=1e-5,
            max_depth=250, # 1000.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_npy_file(self, rel_path):
        npy_path_or_content = os.path.join(self.dataset_dir, rel_path)
        image = np.load(npy_path_or_content).squeeze()
        
        # TODO: CHANGED! Maybe there's a better way to do this?
        # If the image h and w are not divisible by 8, crop the image
        factor = 8
        if image.shape[0] % factor != 0 or image.shape[1] % factor != 0:
            image = image[: image.shape[0] // factor * factor, : image.shape[1] // factor * factor]
        
        data = image[np.newaxis, :, :]
        return data

    def _read_depth_file(self, rel_path):
        depth = self._read_npy_file(rel_path)
        depth[np.isnan(depth)] = self.max_depth
        return depth
