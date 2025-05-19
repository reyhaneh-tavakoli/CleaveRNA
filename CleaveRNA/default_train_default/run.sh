#!/bin/bash
# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Define directories
script_dir="../CleaveRNA.py"  # Correctly reference CleaveRNA.py

# Define the input files and parameters
TARGETS="target_1.fasta target_2.fasta"  # Use space-separated file paths
PARAMS="test_default.csv"
DEFAULT_TRAIN_FILE="HPBC"

# Activate environment with dependencies
conda activate intarna_env

# Record the start time
start_time=$(date)

# Default mode
echo "Running in default mode..."
output_dir=$(pwd)  # Use the current directory where the script is run
python3 "$script_dir" --targets $TARGETS --params $PARAMS --feature_mode default --default_train_file $DEFAULT_TRAIN_FILE --output_dir "$output_dir"

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
