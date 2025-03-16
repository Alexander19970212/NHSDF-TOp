#!/bin/bash

# Script to run VAE_training.py for all configuration files in a directory

# CONFIG_DIR="configs/NN_sdf_experiments/model_arch_15_wMI_wT"
CONFIG_DIR="configs/3D_heaviside_sdf"
DATASET_PATH="shape_datasets/quadrangle_3DHeavisideSDF"
TRAIN_INDEX_LIST_CSV="shape_datasets/quadrangle_3DHeavisideSDF/quadrangle_3DHeavisideSDF_train.csv"
TEST_INDEX_LIST_CSV="shape_datasets/quadrangle_3DHeavisideSDF/quadrangle_3DHeavisideSDF_test.csv"
CONFIG_NAME="3d_HvDecGlobal"
MAX_EPOCHS=1  # Adjust as needed

METRICS_FILE_RECON="src/3d_HvDecGlobal_resEncd.json"

RUN_NAME="run_3d_HvDecGlobal_resEncd"

python train/global_training_3d_via_HvDecoder.py \
    --max_epochs "$MAX_EPOCHS" \
    --dataset_dir "$DATASET_PATH" \
    --train_index_list_csv "$TRAIN_INDEX_LIST_CSV" \
    --test_index_list_csv "$TEST_INDEX_LIST_CSV" \
    --config_dir "$CONFIG_DIR" \
    --config_name "$CONFIG_NAME" \
    --metrics_file "$METRICS_FILE_RECON" \  
    --plot_reconstracted \
    --n_samples 6 \
    --run_name "$RUN_NAME"

