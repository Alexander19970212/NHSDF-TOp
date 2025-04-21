#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

# CONFIG_DIR="configs/NN_sdf_experiments/model_arch_15_wMI_wT"
CONFIG_DIR="configs/NN_sdf_experiments/final_experiments/VAEs"
MAX_EPOCHS=1  # Adjust as needed
DATASET_PATH="shape_datasets"

# ROUND_NUM=6

# for ROUND_NUM in 6 8 9; do

# for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
CONFIG_NAME="VAE_DeepSDF"
# RUN_NAME="Bprec"
RUN_NAME="20smf"
DATASET_TYPE="tripple"

echo "Running VAE_training.py with config: $CONFIG_NAME"

python train/investigate_weight_robustness.py \
    --max_epochs "$MAX_EPOCHS" \
    --dataset_path "$DATASET_PATH" \
    --config_dir "$CONFIG_DIR" \
    --config_name "$CONFIG_NAME" \
    --run_name "$RUN_NAME" \
    --dataset_type "$DATASET_TYPE" \
    --layers_to_perturbate "decoder"

echo "Completed run: $RUN_NAME"
echo "----------------------------------------"

echo "Running VAE_training.py with config: $CONFIG_NAME"

python train/investigate_weight_robustness.py \
    --max_epochs "$MAX_EPOCHS" \
    --dataset_path "$DATASET_PATH" \
    --config_dir "$CONFIG_DIR" \
    --config_name "$CONFIG_NAME" \
    --run_name "$RUN_NAME" \
    --dataset_type "$DATASET_TYPE" \
    --layers_to_perturbate "decoder_residual"

echo "Completed run: $RUN_NAME"
echo "----------------------------------------"

echo "Running VAE_training.py with config: $CONFIG_NAME"

python train/investigate_weight_robustness.py \
    --max_epochs "$MAX_EPOCHS" \
    --dataset_path "$DATASET_PATH" \
    --config_dir "$CONFIG_DIR" \
    --config_name "$CONFIG_NAME" \
    --run_name "$RUN_NAME" \
    --dataset_type "$DATASET_TYPE" \
    --layers_to_perturbate "decoder_output"

echo "Completed run: $RUN_NAME"
echo "----------------------------------------"

echo "Running VAE_training.py with config: $CONFIG_NAME"

python train/investigate_weight_robustness.py \
    --max_epochs "$MAX_EPOCHS" \
    --dataset_path "$DATASET_PATH" \
    --config_dir "$CONFIG_DIR" \
    --config_name "$CONFIG_NAME" \
    --run_name "$RUN_NAME" \
    --dataset_type "$DATASET_TYPE" \
    --layers_to_perturbate "encoder_decoder"

echo "Completed run: $RUN_NAME"
echo "----------------------------------------"
