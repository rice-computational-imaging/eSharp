# Enable debugging (prints out every command)
set -x

# Set to use only specific GPU
export CUDA_VISIBLE_DEVICES=3

export BASE_DATA_DIR=/path/to/data/event3d/  # directory of training data
export BASE_CKPT_DIR=/path/to/models/  # directory of pretrained checkpoint

python train.py \
--config config/train_marigold.yaml \
--output_dir /path/to/models \
--base_ckpt_dir /path/to/models/marigold-v1-0-pretrained \
--resume_run /path/to/models/train_marigold_vae_robust_cosine/checkpoint/latest \
# --ckpt_tag vae_robust_cosine \