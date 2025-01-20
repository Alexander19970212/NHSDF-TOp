#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

# CONFIG_DIR="configs/NN_sdf_experiments/model_arch_minmi"
CONFIG_DIR="configs/NN_sdf_experiments/model_arch_minmi2"
# CONFIG_DIR="configs/NN_sdf_experiments/model_arch"
DATASET_PATH="shape_datasets"
MODEL_DIR="model_weights"
METRICS_FILE="src/reconstruction_metrics.json"
MAX_EPOCHS=1  # Adjust as needed

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    RUN_NAME="run_${CONFIG_NAME}"

    echo "Running AE_reconstructor_training.py with config: $CONFIG_NAME"

    python train/separate_training_reconstructor.py \
        --max_epochs "$MAX_EPOCHS" \
        --dataset_path "$DATASET_PATH" \
        --model_dir "$MODEL_DIR" \
        --config_dir "$CONFIG_DIR" \
        --config_name "$CONFIG_NAME" \
        --metrics_file "$METRICS_FILE"

    echo "Completed run: $RUN_NAME"
    echo "----------------------------------------"
done

echo "All training runs completed."