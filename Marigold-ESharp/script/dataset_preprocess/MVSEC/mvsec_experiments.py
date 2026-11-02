#%%

import numpy as np
import os
import matplotlib.pyplot as plt

import imageio.v3 as io
import h5py

from tqdm import tqdm

#%%

fpng = "/path/to/data/event3d/MVSEC/mvsec_outdoor_day1/depth/frames/frame_0000000011.png"
fnpy = "/path/to/data/event3d/MVSEC/mvsec_outdoor_day1/depth/data/depth_0000000011.npy"

fevent = "/path/to/data/event3d/MVSEC/mvsec_outdoor_day1/events/voxels/event_tensor_0000000010.npy"

# depth frame image (lower res)
img_png = io.imread(fpng)

# depth and disparity from npy
img_npy = np.load(fnpy)
mask_npy = np.isnan(img_npy)
img_npy[mask_npy] = 0

log_disp_npy = np.clip(np.log(1/img_npy), a_min = -5, a_max = 0)
log_disp_npy[mask_npy] = log_disp_npy.min()

plt.figure()
plt.imshow(img_png)
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(log_disp_npy, cmap="magma")
plt.axis("off")
# plt.colorbar()
plt.show()

#%%

# event tensor
events = np.load(fevent)

img1 = events[0,:,:].squeeze()
plt.figure()
plt.imshow(img1)
plt.title(f"min: {img1.min()}, max: {img1.max()}")
plt.colorbar()

img2 = events[1,:,:].squeeze()
plt.figure()
plt.imshow(img2)
plt.title(f"min: {img2.min()}, max: {img2.max()}")
plt.colorbar()

img3 = events[2,:,:].squeeze()
plt.figure()
plt.imshow(img3)
plt.title(f"min: {img3.min()}, max: {img3.max()}")
plt.colorbar()

img4 = events[3,:,:].squeeze()
plt.figure()
plt.imshow(img4)
plt.title(f"min: {img4.min()}, max: {img4.max()}")
plt.colorbar()

img5 = events[4,:,:].squeeze()
plt.figure()
plt.imshow(img5)
plt.title(f"min: {img5.min()}, max: {img5.max()}")
plt.colorbar()

#%%


import numpy as np
import os
import matplotlib.pyplot as plt

import imageio.v3 as io
import h5py

from tqdm import tqdm

# MVSEC data

base_dir = "/path/to/data/event3d/MVSEC/outdoor_day1/"
event_loc = os.path.join(base_dir, "outdoor_day1_data.hdf5")

# read the hdf5 file as a dictionary
data = h5py.File(event_loc)


#%%
images = data['davis']['left']['image_raw']
image_ts = data['davis']['left']['image_raw_ts']