#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

CONFIG_DIR="configs/NN_sdf_experiments/lat_dim"
DATASET_PATH="shape_datasets"
METRICS_FILE="src/metrics_lat_dim2.json"
MAX_EPOCHS=1  # Adjust as needed

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    RUN_NAME="run_${CONFIG_NAME}"

    echo "Running VAE_training.py with config: $CONFIG_NAME"

    python train/VAE_training.py \
        --max_epochs "$MAX_EPOCHS" \
        --dataset_path "$DATASET_PATH" \
        --config_dir "$CONFIG_DIR" \
        --config_name "$CONFIG_NAME" \
        --metrics_file "$METRICS_FILE"

    echo "Completed run: $RUN_NAME"
    echo "----------------------------------------"
done

echo "All training runs completed."