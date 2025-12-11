# Enable debugging (prints out every command)
set -x

# Set to use only specific GPU
export CUDA_VISIBLE_DEVICES=5

export BASE_DATA_DIR=/shared/ad150/event3d/  # directory of training data
export BASE_CKPT_DIR=/shared/ad150/event3d/marigold/checkpoint/  # directory of pretrained checkpoint

python train.py \
--config config/train_marigold_log_disparity.yaml \
--output_dir /shared/ad150/event3d/marigold/checkpoint \
--base_ckpt_dir /shared/ad150/event3d/marigold/checkpoint/marigold-v1-0-pretrained \
--resume_run /shared/ad150/event3d/marigold/checkpoint/train_marigold_log_disparity_vamos/checkpoint/latest \
# --ckpt_tag vamos \