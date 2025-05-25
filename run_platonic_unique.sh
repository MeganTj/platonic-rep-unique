#!/bin/bash

# Define the dataset, subset, and modelset
dataset="minhuh/prh"
subset="wit_1024"
modelset="val"

# Define your base directory
BASE_DIR="/scratch/platonic/results_test"

# Function to handle cleanup when the script is interrupted
cleanup() {
  echo "Interrupt received. Killing background processes..."
  kill $(jobs -p)  # Kill all background jobs
  exit 1  # Exit the script with an error code
}

# Trap SIGINT (Ctrl+C) signal to run the cleanup function
trap cleanup SIGINT

# Loop through different perturbation percentages
for perturbation in {10..100..10}
do
  echo "Processing perturbation $perturbation"

  # Run both the text modality and vision modality extraction in parallel for the current perturbation
  python extract_features.py --dataset $dataset --subset $subset --modelset $modelset --modality language --pool avg --output_dir ${BASE_DIR}/gpt_u=${perturbation}_features --json_perturbations perturbed/data.json --text_percentage_perturbation $perturbation &

  pid_text=$!  # Get the process ID of the text modality process

  python extract_features.py --dataset $dataset --subset $subset --modelset $modelset --modality vision --pool cls --output_dir ${BASE_DIR}/gpt_u=${perturbation}_features --json_perturbations perturbed/data.json --text_percentage_perturbation $perturbation &

  pid_vision=$!  # Get the process ID of the vision modality process

  # Wait for both the text and vision modality processes to finish
  wait $pid_text
  wait $pid_vision

  echo "Completed perturbation $perturbation"
done

# Loop through different perturbation percentages
for perturbation in {10..100..10}
do
  echo "Processing perturbation $perturbation"

  # Run the measure_alignment.py script for the current perturbation
  python measure_alignment.py --dataset $dataset --subset $subset --modelset $modelset \
    --modality_x language --pool_x avg --modality_y vision --pool_y cls \
    --input_dir ${BASE_DIR}/gpt_u=${perturbation}_features \
    --output_dir ${BASE_DIR}/gpt_u=${perturbation}_alignment 

  echo "Completed perturbation $perturbation"
done
