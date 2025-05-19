#!/bin/bash

# Define directories
script_dir="../CleaveRNA.py"

# Define the input files and parameters
TARGETS="target_1.fasta,target_2.fasta"
PARAMS="test_target_check.csv"
DEFAULT_TRAIN_FILE="HPBC"

# activate environment with dependencies
conda activate intarna_env


# Record the start time
start_time=$(date)

# Default mode
echo "Running in default mode..."
python "${script_dir}" --targets $TARGETS --params test_default.csv --feature_mode default --default_train_file $DEFAULT_TRAIN_FILE

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"