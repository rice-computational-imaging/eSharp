#%%

import numpy as np
import glob
import os
from tqdm import tqdm

# Define the root directory where the search should begin
# the starting directory
root_dir = "/path/to/data/event3d/MVSEC/"
# directory where the text files should be saved
text_file_dir = "data_split/mvsec/"
# keyword to search for in the directory names
keyword = "val"  # "train" or "val" or "vis_sample"
encoding = "VAE_ROBUST" # "LINEAR" or "PYRAMIDAL" pr "VAE_ROBUST"

output_file = os.path.join(text_file_dir, f"mvsec_{keyword}.txt")
output_file_small = os.path.join(text_file_dir, f"mvsec_{keyword}_small.txt")

print(f"Writing events to {output_file}")

# Use glob to find all directories with 'train' in their names
train_dirs = [d for d in glob.glob(os.path.join(root_dir, '**', f"*{keyword}*"), recursive=True) if os.path.isdir(d)]

# Initialize the text file (create a new empty file)
with open(output_file, "w") as file:
    pass  # Just creating/clearing the file
with open(output_file_small, "w") as file:
    pass  # Just creating/clearing the file

# This is used to reduce the number of files in the small file
small_factor = 14

print(f"Directories with {keyword}' in their names, total of {len(train_dirs)}")
for dir_path in tqdm(train_dirs, total=len(train_dirs)):
    print(dir_path)
    # get all the event files in the directory
    event_files = glob.glob(os.path.join(dir_path, "encoded_full", encoding, "**", "*.tif"), recursive=True)
    # print the filenames of the event files in a text file
    # append the filenames to the text file
    with open(output_file, "a") as file, open(output_file_small, "a") as file_small:
        for iter, event_file in tqdm(enumerate(event_files), total=len(event_files)):
            # get the corresponding depth file name too
            # get the number of the event file
            event_file_number = event_file.split("/")[-1].split("_")[-1].split(".")[0]
            extension = ".npy"
            depth_files = glob.glob(os.path.join(dir_path, "depth", "data", f"*{event_file_number}*{extension}"))
            depth_file = depth_files[0]

            # Get rid of the root directory in the event file name
            event_file_name = event_file.replace(root_dir, "")
            depth_file_name = depth_file.replace(root_dir, "")

            # Before writing check if the depth file exists
            file1 = root_dir + event_file_name
            file2 = root_dir + depth_file_name

            if (os.path.exists(file1) 
            and os.path.exists(file2)):
                # write the event file name and the depth file name to the text file
                file.write(event_file_name + " " + depth_file_name + "\n")

                if (iter % small_factor == 0 and keyword == "val"):
                    file_small.write(event_file_name + " " + depth_file_name + "\n")

