#!/usr/bin/env bash
set -e
set -x

export BASE_DATA_DIR=/path/to/data/event3d/  # directory of training data
export BASE_CKPT_DIR=/path/to/models/  # directory of pretrained checkpoint

export CUDA_VISIBLE_DEVICES=0

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"/path/to/models/marigold-any-inference"}
subfolder=${2:-"eval"}

python infer.py  \
    --base_data_dir $BASE_DATA_DIR \
    --checkpoint $ckpt \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --processing_res 0 \
    --dataset_config config/dataset/data_carla_test.yaml \
    --inverse_log_plot \
    --color_map magma \
    --output_dir output/${subfolder}/carla_test/prediction \
    --vis_alignment least_square \
    # --vis_alignment inverse_log \
    # --seed 1234 \
