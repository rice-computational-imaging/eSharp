#%%

import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import os

def adjust_file_names(line):
    # Regex pattern to match file names and extract the number
    pattern = r'(LINEAR/event_frame_)(\d+)(\.tif)'

    # Function to subtract 1 from the matched number
    def subtract_one(match):
        base_name = match.group(1)
        file_number = int(match.group(2))
        extension = match.group(3)
        # Subtract 1 and reformat with leading zeros
        new_file_number = file_number - 1
        return f"{base_name}{new_file_number:05d}{extension}"

    # Replace only the first two occurrences of the pattern
    new_line = re.sub(pattern, subtract_one, line, count=2)

    return new_line

def convert_line(line):
    parts = line.split()

    # Comment out the following line for monocular datasets
    # # Skip lines with Camera_1
    # if "Camera_1" in parts[0]:
    #     return None
    
    if (("Scene20/clone/" in parts[0]) or ("Scene20/fog/" in parts[0]) 
    or ("Scene20/morning/" in parts[0]) or ("Scene20/overcast/" in parts[0])
    or ("Scene20/rain/" in parts[0]) or ("Scene20/sunset/" in parts[0]) 
    or ("Scene01/15-deg-left/" in parts[0])
    or ("depth_00000" in parts[1])):
        return None


    # Split RGB and depth file paths
    rgb_path = parts[0] #os.path.join("vkitti2", parts[0])
    depth_path = parts[1] #os.path.join("vkitti2", parts[1])
    
    # Modify the RGB path for Camera_0 and Camera_1
    rgb_new_path_0 = rgb_path.replace("rgb/Camera_0", "rgb/Camera_0_event/LINEAR").replace("rgb/Camera_1", "rgb/Camera_1_event/LINEAR").replace("rgb_", "event_frame_").replace(".jpg", ".tif")
    
    # Return the new formatted line
    line_temp = f"{rgb_new_path_0} {depth_path}"
    out_line = adjust_file_names(line_temp)

    parts_out = out_line.split()
    base_dir = "/path/to/data/event3d/vkitti2/"

    file1 = base_dir + parts_out[0]
    file2 = base_dir + parts_out[1]

    # print(file1, file2)

    if (os.path.exists(file1) 
        and os.path.exists(file2)):
        return out_line
    else:
        return None

    return out_line

small_factor = 100

def process_file(input_file, output_file):
    counter = 0
    with (open(input_file, 'r') as infile, open(output_file, 'w') as outfile, 
          open(output_file.replace(".txt", "_small.txt"), 'w') as outfile_small):
        for line in infile:
            new_line = convert_line(line.strip())
            if new_line is not None:
                outfile.write(new_line + '\n')
                if (counter % small_factor == 0 and "val" in input_file):
                    outfile_small.write(new_line + '\n')
            counter += 1
#%%
# Validation dataset files
input_file = "../../../data_split/vkitti/vkitti_val_orig.txt"  # Replace with your input file path
output_file = "../../../data_split/vkitti/vkitti_val.txt"  # Replace with your output file path
process_file(input_file, output_file)

#%%
# Train dataset files
input_file = "../../../data_split/vkitti/vkitti_train_orig.txt"  # Replace with your input file path
output_file = "../../../data_split/vkitti/vkitti_train.txt"  # Replace with your output file path
process_file(input_file, output_file)