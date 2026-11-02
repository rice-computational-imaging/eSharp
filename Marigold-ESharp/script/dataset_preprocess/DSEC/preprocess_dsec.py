#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio as io

def get_event_image(fname, time_per_frame, first_frame, total_frames):
    # load events
    data = h5py.File(fname, "r")
    group = data["events"]
    p_arr = np.array(group["p"])
    t_arr = np.array(group["t"])
    x_arr = np.array(group["x"])
    y_arr = np.array(group["y"])

    print(p_arr.shape, t_arr.shape, x_arr.shape, y_arr.shape)

    actual_time_1 = first_frame * time_per_frame
    actual_time_2 = actual_time_1 + total_frames * time_per_frame

    all_ts = np.argwhere(np.logical_and(t_arr > actual_time_1, t_arr < actual_time_2))
    tsi1 = int(all_ts[0])
    tsi2 = int(all_ts[-1])
    print(tsi1, tsi2)

    # Take some events from a "frame"
    print(t_arr[tsi1:tsi2].shape)

    t_frame = t_arr[tsi1:tsi2]
    x_frame = x_arr[tsi1:tsi2]
    y_frame = y_arr[tsi1:tsi2]
    p_frame = p_arr[tsi1:tsi2]

    # # Plot events
    # plt.figure(figsize=(10, 5))
    # plt.scatter(x_frame, y_frame, c=t_frame, s=1, cmap='viridis')
    # plt.colorbar(label='Timestamp')
    # plt.gca().invert_yaxis()
    # plt.xlabel('X')
    # plt.ylabel('Y')


    img = np.zeros([480, 640])+128
    for i in range(len(t_frame)):
        img[y_frame[i], x_frame[i]] = 255*p_frame[i]

    img = np.uint8(np.repeat(img[:,:,np.newaxis], 3, axis=-1))
    print(img.shape)

    return img


#%%

s = 1e6
ms = 1e3
us = 1

fps = 30
time_per_frame = 1*s/fps
first_frame = 20
total_frames = 20


fname_left = "/path/to/data/event_camera/test/interlaken_00_a/events/left/events.h5"
fname_right = "/path/to/data/event_camera/test/interlaken_00_a/events/right/events.h5"

actual_time_1 = 0


img_left = get_event_image(fname_left, time_per_frame, first_frame, total_frames)
img_right = get_event_image(fname_right, time_per_frame, first_frame, total_frames)

plt.figure()
plt.imshow(img_left, cmap='gray')
plt.colorbar()

plt.figure()
plt.imshow(img_right, cmap='gray')
plt.colorbar()

sname_left = f"script/input/events/DSEC_events_{first_frame}_left.tif"
sname_right = f"script/input/events/DSEC_events_{first_frame}_right.tif"

io.imwrite(sname_left, img_left)
io.imwrite(sname_right, img_right)

# %%
