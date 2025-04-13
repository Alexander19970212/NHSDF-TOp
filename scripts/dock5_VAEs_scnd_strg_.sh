#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

# CONFIG_DIR="configs/NN_sdf_experiments/model_arch_15_wMI_wT"
CONFIG_DIR="configs/NN_sdf_experiments/final_experiments/VAEs"
DATASET_PATH="shape_datasets"
MAX_EPOCHS=1  # Adjust as needed
# ROUND_NUM=6

for ROUND_NUM in 6 8 9; do
    echo "Running round ${ROUND_NUM}"
    METRICS_FILE="src/final_metrics_round${ROUND_NUM}_scnd_strtg_VAEs.json"
    METRICS_FILE_RECON="src/final_metrics_round${ROUND_NUM}_scnd_strtg_VAEs_recon.json"


    for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
        CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
        RUN_NAME="run_${CONFIG_NAME}"

        echo "Running VAE_training.py with config: $CONFIG_NAME"

        python train/global_training_via_reconstructor.py \
            --max_epochs "$MAX_EPOCHS" \
            --dataset_path "$DATASET_PATH" \
            --config_dir "$CONFIG_DIR" \
            --config_name "$CONFIG_NAME" \
            --metrics_file "$METRICS_FILE_RECON"

        echo "Completed run: $RUN_NAME"
        echo "----------------------------------------"
    done

    for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
        CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
        RUN_NAME="run_${CONFIG_NAME}"

        echo "Running VAE_training.py with config: $CONFIG_NAME"

        python train/separate_training_HvDecoder.py \
            --max_epochs "$MAX_EPOCHS" \
            --dataset_path "$DATASET_PATH" \
            --config_dir "$CONFIG_DIR" \
            --config_name "$CONFIG_NAME" \
            --metrics_file "$METRICS_FILE"

        echo "Completed run: $RUN_NAME"
        echo "----------------------------------------"
    done
done

echo "All training runs completed."