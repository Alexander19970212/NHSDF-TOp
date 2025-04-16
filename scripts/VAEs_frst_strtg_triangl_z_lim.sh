#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

CONFIG_DIR="configs/NN_sdf_experiments/final_experiments/VAEs"
MAX_EPOCHS=1  # Adjust as needed
DATASET_PATH="shape_datasets"
MODELS_DIR="model_weights"

CONFIG_NAME="VAE_DeepSDF_triangle"
RUN_NAME="Bprec"
DATASET_TYPE="triangle"

echo "Running save_z_limits.py with config: $CONFIG_NAME"

python train/save_z_limits.py \
    --dataset_path "$DATASET_PATH" \
    --configs_dir "$CONFIG_DIR" \
    --models_dir "$MODELS_DIR" \
    --config_name "$CONFIG_NAME" \
    --run_name "$RUN_NAME" \
    --dataset_type "$DATASET_TYPE"

echo "Completed run: $RUN_NAME"
echo "----------------------------------------"

echo "All training runs completed."