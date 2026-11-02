# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
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
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import matplotlib
import matplotlib.pyplot
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="magma", valid_mask=None, inverse_log_plot=False
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    if (inverse_log_plot == True):
        depth = convert_to_inverse_log_magma(depth, amax=25000, amin=5)
    else:
        depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def convert_to_inverse_log_magma(imgs, amax=25000, amin=5):
    # current shape is [(B,), H, W]
    img_temp = (imgs * 65535.0/256)#.astype(np.uint16)
    img_temp = np.clip(img_temp, a_min = amin, a_max=amax)
    img_temp = np.log(1/img_temp)
    
    # Take channel wise min and max
    img_temp_min = np.min(img_temp, axis=(1,2), keepdims=True)
    img_temp_max = np.max(img_temp, axis=(1,2), keepdims=True)

    img_temp = (img_temp - img_temp_min)/np.abs(img_temp_max - img_temp_min)

    return img_temp


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST_EXACT,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method

def get_edge_map(imgs, edge_threshold=0.6, edge_type="sobel"):
    """
    Get edge map from image.
    """

    def sobel_edge_detector(image):
        # Define Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

        # Move kernels to the same device as image
        device = image.device
        sobel_x, sobel_y = sobel_x.to(device), sobel_y.to(device)

        # Apply Sobel filters
        grad_x = F.conv2d(image, sobel_x, padding=1).to(device)
        grad_y = F.conv2d(image, sobel_y, padding=1).to(device)

        # Combine gradients
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Threshold edges using sigmoid
        scale = 10.0
        edges = torch.sigmoid(scale * (edges - edge_threshold))

        return edges


    if edge_type == "sobel":
        edge_map = sobel_edge_detector(imgs)
    else:
        raise NotImplementedError
    return edge_map


def box_filter(x, r):
    # x: (B, C, H, W)
    # r: scalar
    if (len(x.shape) == 3):
        x = x.unsqueeze(0)
    elif (len(x.shape) == 2):
        x = x.unsqueeze(0).unsqueeze(0)

    B, C, H, W = x.shape
    print(x.shape)
    # write a code using conv2d with a kernel of size 2r+1

    device = x.device
    conv = torch.nn.Conv2d(C, C, kernel_size=2*r+1, padding=r, bias=False).to(device)
    print(conv.weight.data.shape)
    conv.weight.data = torch.ones_like(conv.weight.data)

    return conv(x)


'''
#%%
# TODO: Used for testing the above code, delete it later

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')  

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

# Convert to tensor
china_tensor = torch.tensor(china, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
china_tensor2 = transform(Image.fromarray(china))
china_tensor2.requires_grad = True
# Display the image
plt.figure()
plt.imshow(china_tensor.squeeze().permute(1, 2, 0))
# plt.imshow(china_tensor2.squeeze().detach().numpy())
plt.axis('off')
plt.show()

# Get edge map
china_edge_map = get_edge_map(china_tensor2, edge_threshold=0.9)
plt.figure()
plt.imshow(china_edge_map.squeeze().detach().numpy())
plt.colorbar()
plt.axis('off')
plt.show()

print(china_tensor2.requires_grad)
print(china_edge_map.requires_grad)
'''