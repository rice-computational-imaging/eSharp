#!/bin/bash
set -e  # Exit if any command fails
set -x  # Show commands as they are run

# Define the super directory where all scenes are located
super_dir="/path/to/data/event3d/vkitti2"

# Loop through each frames directory found within the super directory
find "$super_dir" -type d -name "frames" | while read -r base_dir; do

    # Define event_dir relative to base_dir
    event_dir="rgb/Camera_0_event/"
    # Define save_dir based on the base_dir path
    save_dir="${base_dir}/${event_dir}"

    # Define the path to the required events.txt file
    events_file="${base_dir}/${event_dir}/events.txt"
    # Check if events.txt exists, if not, skip this iteration
    if [[ ! -f "$events_file" ]]; then
        echo "Skipping $base_dir as events.txt is missing."
        continue
    fi

    # # Run the Python script with these parameters
    python script/dataset_preprocess/vkitti/generate_event_dataset.py \
        --base_dir "$base_dir" \
        --event_dir "$event_dir" \
        --save_dir "$save_dir" \
        --time_encoding "PYRAMIDAL" \
        # --vmax 0.04 \

    echo "save_dir: $save_dir"


    # Define event_dir relative to base_dir
    event_dir="rgb/Camera_1_event/"
    # Define save_dir based on the base_dir path
    save_dir="${base_dir}/${event_dir}"

    # Define the path to the required events.txt file
    events_file="${base_dir}/${event_dir}/events.txt"
    # Check if events.txt exists, if not, skip this iteration
    if [[ ! -f "$events_file" ]]; then
        echo "Skipping $base_dir as events.txt is missing."
        continue
    fi

    # # Run the Python script with these parameters
    python script/dataset_preprocess/vkitti/generate_event_dataset.py \
        --base_dir "$base_dir" \
        --event_dir "$event_dir" \
        --save_dir "$save_dir" \
        --time_encoding "PYRAMIDAL" \
        # --vmax 0.04 \

    echo "save_dir: $save_dir"

done