#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

CONFIG_DIR="configs/NN_sdf_experiments/final_experiments"
MAX_EPOCHS=1  # Adjust as needed
DATASET_PATH="shape_datasets"
MODELS_DIR="model_weights"


for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    RUN_NAME="run_${CONFIG_NAME}"

    echo "Running save_z_limits.py with config: $CONFIG_NAME"

    python train/save_z_limits.py \
        --dataset_path "$DATASET_PATH" \
        --configs_dir "$CONFIG_DIR" \
        --models_dir "$MODELS_DIR"

    echo "Completed run: $RUN_NAME"
    echo "----------------------------------------"
done

echo "All training runs completed."