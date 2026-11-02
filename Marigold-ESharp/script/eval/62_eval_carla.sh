#!/usr/bin/env bash
set -e
set -x

export BASE_DATA_DIR=/path/to/data/event3d/  # directory of training data
export BASE_CKPT_DIR=/path/to/models/  # directory of pretrained checkpoint

export CUDA_VISIBLE_DEVICES=0

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_carla_test.yaml \
    --prediction_dir output/${subfolder}/carla_test/prediction \
    --output_dir output/${subfolder}/carla_test/test_metric \
    --alignment least_square \
    # --alignment absolute \
