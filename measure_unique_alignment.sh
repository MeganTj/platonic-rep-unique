#!/bin/bash

# Define the dataset, subset, and modelset
dataset="minhuh/prh"
subset="wit_1024"
modelset="val"

# Loop through different perturbation percentages
for perturbation in {5..50..5}
do
  echo "Processing perturbation $perturbation"

  # Run the measure_alignment.py script for the current perturbation
  python measure_alignment.py --dataset $dataset --subset $subset --modelset $modelset \
    --modality_x language --pool_x avg --modality_y vision --pool_y cls \
    --input_dir /scratch/platonic/results/u=${perturbation}_features \
    --output_dir /scratch/platonic/results/u=${perturbation}_alignment

  echo "Completed perturbation $perturbation"
done
