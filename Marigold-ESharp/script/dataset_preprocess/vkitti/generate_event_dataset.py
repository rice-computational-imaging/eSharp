# Use this scrip to generate a dataset of events using 
# the already generated events from any events.txt of [t, x, y, p] format

#######################
# Sample usage of the code
# python generate_event_dataset.py 
# --base_dir="/path/to/user/git/Marigold/script/input/" 
# --event_dir="Camera_events" 
# --npy 
# --vmin=0 --vmax=0.04 --interval_vmax=0.04
# --time_encoding=LINEAR 
# --save_dir="encoded_full"
#######################

import numpy as np
from tqdm import tqdm
import csv
import os
import sys
import imageio.v3 as io
import pandas as pd

from scipy.ndimage import gaussian_filter

import glob
import argparse
import configparser


# Define enums for time encoding
TIME_ENCODING = {
    "LINEAR": 0,
    "LOG": 1,
    "POSITIONAL": 2,
    "PYRAMIDAL": 3,
    "VAE_ROBUST": 4,
}

def normalize(x, relative_vmin=None, relative_vmax=None, interval_vmax=None):
    vmax = x.max()
    vmin = x.min()
    if (relative_vmax is not None):
        vmax = relative_vmax + vmin
    if (relative_vmin is not None):
        vmin = relative_vmin + vmin
    if (interval_vmax is None):
        interval_vmax = vmax - vmin


    # Keep only the values between vmin and vmax
    x = x * (x >= vmin) * (x <= vmax)

    return (x - vmin) / interval_vmax

def positional_encoding_1d(times, num_encodings=3):
    # Positional encoding formula: sin/cos of different frequencies
    encodings = np.zeros((len(times), num_encodings * 2))
    for i in range(num_encodings):
        frequency = 1 / (10000 ** (2 * i / num_encodings))
        encodings[:, 2 * i] = np.sin(times * frequency)
        encodings[:, 2 * i + 1] = np.cos(times * frequency)
    
    # Sum across the encoding dimension to reduce to 1D
    reduced_encoding = np.sum(encodings, axis=1)
    return reduced_encoding


def get_pyramidal_encoding(times, polarity, x, y):
    # Split the events into 3 images

    max_time = times.max()

    # L0 should have all the events - so keep it as it is
    times_L0 = times
    polarity_L0 = polarity
    L0 = {}
    L0["times"] = times_L0
    L0["polarity"] = polarity_L0
    L0["x"] = x
    L0["y"] = y

    # L1 should have events starting from 0.5 times max to max
    indices1 = times >= 0.5 * max_time
    times_L1 = times[indices1]
    polarity_L1 = polarity[indices1]
    x_L1 = x[indices1]
    y_L1 = y[indices1]
    L1 = {}
    L1["times"] = times_L1
    L1["polarity"] = polarity_L1
    L1["x"] = x_L1
    L1["y"] = y_L1

    # L2 should have events starting from 0.75 times max to max
    indices2 = times >= 0.75 * max_time
    times_L2 = times[indices2]
    polarity_L2 = polarity[indices2]
    x_L2 = x[indices2]
    y_L2 = y[indices2]
    L2 = {}
    L2["times"] = times_L2
    L2["polarity"] = polarity_L2
    L2["x"] = x_L2
    L2["y"] = y_L2

    return L0, L1, L2


def get_vae_robust_encoding(times, polarity, x, y, image):
    # Split the events into 3 images

    max_time = times.max()

    # L0 should have all the events - so keep it as it is
    t = times
    pol = polarity
    xx = x
    yy = y
    neg_p = polarity.min()
    image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), :] = (np.ones_like(t))[pol == neg_p][:, None]
    image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), :]         = (np.zeros_like(t))[pol == 1][:, None]
    image[:,:,:] += gaussian_filter(image[:,:,:], sigma=3.0)

    # L1 should have events starting from 0.5 times max to max
    indices1 = times >= 0.5 * max_time
    t = times[indices1]
    pol = polarity[indices1]
    xx = x[indices1]
    yy = y[indices1]
    image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), :] = (np.ones_like(t))[pol == neg_p][:, None]
    image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), :]         = (np.zeros_like(t))[pol == 1][:, None]
    image[:,:,:] += gaussian_filter(image[:,:,:], sigma=2.0)

    # L2 should have events starting from 0.75 times max to max
    indices2 = times >= 0.75 * max_time
    t = times[indices2]
    pol = polarity[indices2]
    xx = x[indices2]
    yy = y[indices2]
    image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), :] = (np.ones_like(t))[pol == neg_p][:, None]
    image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), :]         = (np.zeros_like(t))[pol == 1][:, None]
    image[:,:,:] += gaussian_filter(image[:,:,:], sigma=1.0)

    image = (image - image.min()) / (image.max() - image.min())

    # image = image * 2 - 1
    # image = np.sign(image)
    # image = (image + 1) / 2

    return image


def save_event_frames(base_dir, event_dir, save_dir, time_encoding, args):

    height = args.get("height", None)
    width = args.get("width", None)
    if (height is None or width is None):
        # get the height and width from the first image
        # Get the first image in the directory
        if (args.get("npy")):
            print("Reading from npy files")
            image_files = glob.glob(os.path.join(base_dir, "../depth/frames/", "*.png"))
        else:
            print("Reading from txt events")
            image_files = glob.glob(os.path.join(base_dir, "depth/Camera_0/", "*.png"))
        if len(image_files) == 0:
            image_files = glob.glob(os.path.join(base_dir, event_dir, "*.tif"))
            if len(image_files) == 0:
                raise FileNotFoundError(f"No images found in the directory {os.path.join(base_dir, event_dir)}")
        # Read the first image to get the height and width
        img = io.imread(image_files[0])
        height = img.shape[0]
        width = img.shape[1]

    if args.get("npy"):
        save_from_discrete_event_frames(base_dir, event_dir, save_dir, time_encoding, height, width, args)
    else:
        save_from_continous_events(base_dir, event_dir, save_dir, time_encoding, height, width, args)

def save_from_discrete_event_frames(base_dir, event_dir, save_dir, time_encoding, height, width, args):
    
    # Get all the npy files in the directory
    npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npy")))
    if len(npy_files) == 0:
        print("No npy files found. Trying to load npz files")
        npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npz")))
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No npy/npz files found in the directory {os.path.join(base_dir, event_dir)}")
    
    images = []
    for npy_file in tqdm(npy_files, desc="Processing npy/npz files", total=len(npy_files), 
                         dynamic_ncols=True, ascii=True):
        # Load the npz file
        try:
            data = np.load(npy_file)
            times = data[:,0].squeeze()/1e6
            x = data[:,1].squeeze()
            y = data[:,2].squeeze()
            polarity = data[:,3].squeeze()
        except:
            data = np.load(npy_file)
            times = data["t"].squeeze()/1e6
            x = data["x"].squeeze()
            y = data["y"].squeeze()
            polarity = data["p"].squeeze()

        vmin = args.get("vmin", None)
        vmax = args.get("vmax", None)
        interval_vmax = args.get("interval_vmax", None)

        # First encode time as color
        if TIME_ENCODING["LINEAR"] == time_encoding:
            # Normalize the time to be between 0 and 1
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
        elif TIME_ENCODING["LOG"] == time_encoding:
            # TODO: Not implemented properly
            # Normalize the time to be between 0 and 1
            times = normalize(np.log(times+1), interval_vmax=vmax)
        elif TIME_ENCODING["POSITIONAL"] == time_encoding:
            times = positional_encoding_1d(times)  
        elif (TIME_ENCODING["PYRAMIDAL"] == time_encoding):
            # Normalize the time to be between 0 and 1
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
            L0, L1, L2 = get_pyramidal_encoding(times, polarity, x, y)
        elif (TIME_ENCODING["VAE_ROBUST"] == time_encoding):
            # Normalize the time to be between 0 and 1
            image = np.zeros((height, width, 3), dtype=np.float32) + 0.5
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
            event_image = get_vae_robust_encoding(times, polarity, x, y, image)


        # Initialize the image as all ones
        if (TIME_ENCODING["PYRAMIDAL"] == time_encoding):
            image = np.zeros((height, width, 3))
            image = image + 0.5

            neg_p = polarity.min()

            xx = L0["x"]
            yy = L0["y"]
            pol = L0["polarity"]
            t = L0["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 2] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 2]         = (np.ones_like(t))[pol == 1]

            xx = L1["x"]
            yy = L1["y"]
            pol = L1["polarity"]
            t = L1["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 1] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 1]         = (np.ones_like(t))[pol == 1]

            xx = L2["x"]
            yy = L2["y"]
            pol = L2["polarity"]
            t = L2["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 0] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 0]         = (np.ones_like(t))[pol == 1]
        elif (TIME_ENCODING["VAE_ROBUST"] == time_encoding):
            image = event_image
        else:
            image = np.zeros((height, width, 3))
            # Assign the red channel to the 0 polarity events, and make its blue and green channels zero
            times = times
            neg_p = polarity.min()
            image[np.int64(y[polarity == neg_p]), np.int64(x[polarity == neg_p]), 0] = times[polarity == neg_p]
            # Assign the blue channel to the 1 polarity events
            image[np.int64(y[polarity == 1]), np.int64(x[polarity == 1]), 2] = times[polarity == 1]

        # ones = np.ones_like(polarity)
        # image = np.ones((height, width, 3))
        # image[np.int64(y[polarity == -1]), np.int64(x[polarity == -1]), 0] = ones[polarity == -1]
        # image[np.int64(y[polarity == 1]), np.int64(x[polarity == 1]), 2] = ones[polarity == 1]

        images.append(image)

    # Save the images to disk
    total_images = len(images)
    print(f"Saving {total_images} images to disk")

    # Check first if the directory exists
    save_dir = os.path.join(save_dir, args.get("time_encoding", "LINEAR"))
    print(f"Saving images to {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for iter in tqdm(range(total_images), desc="Saving images", 
                     dynamic_ncols=True, ascii=True):
        
        img = np.uint8(255*images[iter])
        savename = os.path.join(save_dir, f"event_frame_{iter:04d}.tif")
        io.imwrite(savename, img)

    # Also write the configuration file
    config = configparser.ConfigParser()
    config["EventDataset"] = {
        "height": str(height),
        "width": str(width),
        "time_encoding": args.get("time_encoding", "LINEAR"),
        "vmin": str(vmin),
        "vmax": str(vmax),
        "interval_vmax": str(interval_vmax),
    }
    # Write the configuration file
    with open(os.path.join(save_dir, "config.ini"), 'w') as configfile:
        config.write(configfile)          


def save_from_continous_events(base_dir, event_dir, save_dir, time_encoding, height, width, args):
    print(f"--- Reading events from {os.path.join(base_dir, event_dir)}")

    filepath = os.path.join(base_dir, event_dir, "events.txt")
    dtypes = {"time": "float32", "x": "int32", "y": "int32", "polarity": "int8"}
    df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None, names=["time", "x", "y", "polarity"], dtype=dtypes)

    times = df["time"].values
    x = df["x"].values
    y = df["y"].values
    polarity = df["polarity"].values

    print(f"--- DONE --- Number of events: {len(times)}")

    # Divide the timestamps according to the mentioned frame times
    filename_frame_times = os.path.join(base_dir, event_dir, "dvs-video-frame_times.txt")

    # Read the frame times from the file
    frame_times = []
    with open(filename_frame_times, 'r') as file:
        for line in file:
            # Skip lines that start with a comment ('#')
            if line.startswith('#') or line.strip() == '':
                continue
            # Each line is of the format "frame_number timestamp"
            frame_times.append(float(line.split()[1]))

    print(f"Number of frames: {len(frame_times)}")

    frame_events = []
    # Use np.searchsorted to find the starting indices for each frame time
    # left  : a[i-1] < v <= a[i]
    start_indices = np.searchsorted(times, frame_times, side='left')
    
    # Iterate through frame times and slice based on the precomputed indices
    for i, frame_time in tqdm(enumerate(frame_times[:-1]), desc="Dividing events into frames", 
                              total=len(frame_times), dynamic_ncols=True, ascii=True):
    
        # Extract the events that belong to this frame
        start = start_indices[i]
        end = start_indices[i+1]
        frame_events.append((times[start:end], x[start:end], y[start:end], polarity[start:end]))

    images = []

    for i, (times, x, y, polarity) in tqdm(enumerate(frame_events), desc="Creating images", 
                                           total=len(frame_events), dynamic_ncols=True, ascii=True):

        vmin = args.get("vmin", None)
        vmax = args.get("vmax", None)
        interval_vmax = args.get("interval_vmax", None)

        # First encode time as color
        if TIME_ENCODING["LINEAR"] == time_encoding:
            # Normalize the time to be between 0 and 1
            # times = normalize(times, interval_vmax=vmax)
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
        elif TIME_ENCODING["LOG"] == time_encoding:
            # TODO: Not implemented properly
            # Normalize the time to be between 0 and 1
            times = normalize(np.log(times+1), interval_vmax=vmax)
        elif TIME_ENCODING["POSITIONAL"] == time_encoding:
            times = positional_encoding_1d(times)
        elif (TIME_ENCODING["PYRAMIDAL"] == time_encoding):
            # Normalize the time to be between 0 and 1
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
            L0, L1, L2 = get_pyramidal_encoding(times, polarity, x, y)

        # Initialize the image as all ones
        if (TIME_ENCODING["PYRAMIDAL"] == time_encoding):
            image = np.zeros((height, width, 3))
            image = image + 0.5

            neg_p = polarity.min()

            xx = L0["x"]
            yy = L0["y"]
            pol = L0["polarity"]
            t = L0["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 2] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 2]         = (np.ones_like(t))[pol == 1]

            xx = L1["x"]
            yy = L1["y"]
            pol = L1["polarity"]
            t = L1["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 1] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 1]         = (np.ones_like(t))[pol == 1]

            xx = L2["x"]
            yy = L2["y"]
            pol = L2["polarity"]
            t = L2["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 0] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 0]         = (np.ones_like(t))[pol == 1]
        else:
            image = np.zeros((height, width, 3))
            # Assign the red channel to the 0 polarity events, and make its blue and green channels zero
            times = times
            neg_p = polarity.min()
            image[np.int64(y[polarity == neg_p]), np.int64(x[polarity == neg_p]), 0] = times[polarity == neg_p]
            # Assign the blue channel to the 1 polarity events
            image[np.int64(y[polarity == 1]), np.int64(x[polarity == 1]), 2] = times[polarity == 1]


        images.append(image)

    # Save the images to disk
    total_images = len(images)
    print(f"Saving {total_images} images to disk")

    # Check first if the directory exists
    save_dir = os.path.join(save_dir, args.get("time_encoding", "LINEAR"))
    print(f"Saving images to {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for iter in tqdm(range(total_images), desc="Saving images", 
                     dynamic_ncols=True, ascii=True):
        
        img = np.uint8(255*images[iter])
        savename = os.path.join(save_dir, f"event_frame_{iter+1:05d}.tif")
        io.imwrite(savename, img)

    # Also write the configuration file
    config = configparser.ConfigParser()
    config["EventDataset"] = {
        "height": str(height),
        "width": str(width),
        "time_encoding": args.get("time_encoding", "LINEAR"),
        # "vmin": str(vmin),
        # "vmax": str(vmax),
    }
    # Write the configuration file
    with open(os.path.join(save_dir, "config.ini"), 'w') as configfile:
        config.write(configfile)

def main(args):

    base_dir = args.get("base_dir")
    time_encoding = TIME_ENCODING[args.get("time_encoding", "LINEAR")]

    event_dir = args.get("event_dir", "rgb/Camera_0_event/")

    save_dir = args.get("save_dir", "encoded_event_frames")
    save_dir = os.path.join(base_dir, event_dir, save_dir)
    print(f"Saving encoded event frames to {save_dir}")

    save_event_frames(base_dir, event_dir, save_dir, time_encoding, args)



def get_args():
    # parse arguments
    parser = argparse.ArgumentParser(description='Generate event dataset')
    parser.add_argument('--base_dir', type=str, 
                        help='Parent directory that houses the '
                        'events.txt or npy files within its subdirectory',
                        required=True)
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save the encoded event frames',
                        default="encoded_event_frames")
    parser.add_argument('--event_dir', type=str, default="rgb/Camera_0_event/",
                        help='Directory containing the event images')
    # Are the events in npy format?
    parser.add_argument('--npy', action='store_true',
                        help='Whether the events are in npy format')
    
    parser.add_argument('--height', type=int, default=None,
                        help='Height of the output image')
    parser.add_argument('--width', type=int, default=None,
                        help='Width of the output image')
    
    parser.add_argument('--time_encoding', type=str, default="LINEAR",
                        help='Time encoding to use for the events')
    
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum value to keep in the event set')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum value to keep in the event set')
    parser.add_argument('--interval_vmax', type=float, default=None,
                        help='interval time to normalize the time to')

    return vars(parser.parse_args())
    


if __name__ == "__main__":
    args = get_args()
    main(args)