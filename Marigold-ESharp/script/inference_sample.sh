# Enable debugging (prints out every command)
set -x

# Set to use only specific GPU
export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1

python run.py \
    --checkpoint /path/to/models/marigold-any-inference \
    --denoise_steps 1 \
    --processing_res 0\
    --input_rgb_dir script/input/events \
    --inverse_log_plot \
    --color_map magma \
    --output_dir script/output/events/edge-loss-2000-ckpt/ \
    --ensemble_size 1 \
    # --guided_steps 5 \