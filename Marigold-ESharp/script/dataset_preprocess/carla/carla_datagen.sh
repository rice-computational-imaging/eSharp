# base_dir="/path/to/user/git/Marigold/script/input/" 
# # base_dir="/script/input/"
# event_dir="Camera_events" 
# # event_dir="Camera_events/"
# save_dir="encoded_full"

# python script/dataset_preprocess/vkitti/generate_event_dataset.py \
# --base_dir "$base_dir" \
# --event_dir "$event_dir" \
# --save_dir "$save_dir" \
# --npy \
# --time_encoding "VAE_ROBUST" \


# Define the super directory where all scenes are located
super_dir="/path/to/data/event3d/carla/"

# Loop through each frames directory found within the super directory
find "$super_dir" -type d -name "events" | while read -r base_dir; do

    # Define event_dir relative to base_dir
    event_dir="data/"
    # Define save_dir based on the base_dir path
    save_dir="${base_dir}/${event_dir}/../frames_event/"

    # Define the path to the required events.txt file
    events_file="${base_dir}/${event_dir}/boundary_timestamps.txt"
    # Check if events.txt exists, if not, skip this iteration
    if [[ ! -f "$events_file" ]]; then
        echo "Skipping $base_dir as boundary_timestamps.txt is missing."
        continue
    fi

    # # Run the Python script with these parameters
    python script/dataset_preprocess/vkitti/generate_event_dataset.py \
        --base_dir "$base_dir" \
        --event_dir "$event_dir" \
        --save_dir "$save_dir" \
        --npy \
        --time_encoding "VAE_ROBUST" \

    echo "save_dir: $save_dir"

done