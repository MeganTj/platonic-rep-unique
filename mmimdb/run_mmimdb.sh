#!/usr/bin/env bash
# run_mmimdb.sh

# Exit if any command fails
set -e

# Define your base directory
BASE_DIR="/scratch/megantj/projects/platonic-rep/mmimdb/test_results2"

# Define your CUDA devices
CUDA_DEVICES="0"

echo "=== Extracting Features ==="
# CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python mmimdb/extract_features.py \
#     --batch_size 2 \
#     --feat_save_dir "${BASE_DIR}/mmimdb_feat"

echo "=== Training Linear Model ==="
# CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python mmimdb/train_linear.py \
#     --input_dir "${BASE_DIR}/mmimdb_feat" \
#     --save_dir  "${BASE_DIR}/mmimdb_performance"

echo "=== Measuring Alignment ==="
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python mmimdb/measure_alignment_mmimdb.py \
    --feat_save_dir "${BASE_DIR}/mmimdb_feat" \
    --output_dir "${BASE_DIR}/mmimdb_align"

echo "=== Plotting Alignment Results ==="
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python mmimdb/plot_alignment.py \
    --perf_dir  "${BASE_DIR}/mmimdb_performance" \
    --align_dir "${BASE_DIR}/mmimdb_align" \
    --plot_dir  "./mmimdb/test_plots"

echo "All steps completed!"
