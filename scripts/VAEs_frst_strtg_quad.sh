#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

# CONFIG_DIR="configs/NN_sdf_experiments/model_arch_15_wMI_wT"
CONFIG_DIR="configs/NN_sdf_experiments/final_experiments/VAEs"
MAX_EPOCHS=3  # Adjust as needed
DATASET_PATH="shape_datasets"

# ROUND_NUM=6

# for ROUND_NUM in 6 8 9; do
METRICS_FILE="src/final_metrics_frst_strtg_VAEs_quad.json"
METRICS_FILE_RECON="src/final_metrics_frst_strtg_VAEs_quad_recon.json"


# for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
CONFIG_NAME="VAE_DeepSDF_quad.yaml"
RUN_NAME="quad"
DATASET_TYPE="quadrangle"

echo "Running VAE_training.py with config: $CONFIG_NAME"

python train/global_training_via_HvDecoder.py \
    --max_epochs "$MAX_EPOCHS" \
    --dataset_path "$DATASET_PATH" \
    --config_dir "$CONFIG_DIR" \
    --config_name "$CONFIG_NAME" \
    --metrics_file "$METRICS_FILE" \
    --run_name "$RUN_NAME" \
    --dataset_type "$DATASET_TYPE"

echo "Completed run: $RUN_NAME"
echo "----------------------------------------"
# done

# for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
# CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
# RUN_NAME="run_${CONFIG_NAME}"

echo "Running VAE_training.py with config: $CONFIG_NAME"

python train/separate_training_reconstructor.py \
    --max_epochs "$MAX_EPOCHS" \
    --dataset_path "$DATASET_PATH" \
    --config_dir "$CONFIG_DIR" \
    --config_name "$CONFIG_NAME" \
    --metrics_file "$METRICS_FILE_RECON" \
    --run_name "$RUN_NAME" \
    --dataset_type "$DATASET_TYPE"

echo "Completed run: $RUN_NAME"
echo "----------------------------------------"
# done