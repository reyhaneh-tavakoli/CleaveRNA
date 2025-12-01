#!/bin/bash
# Initialize conda for bash shell
__conda_setup="$('/home/reytakop/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    export PATH="/home/reytakop/miniconda3/bin:$PATH"
fi
unset __conda_setup

conda activate intarna-env

# Define directories
script_path="../../../../CleaveRNA/CleaveRNA.py"  # Path to CleaveRNA.py

# Define the input files and parameters
TARGET_FILES="target_1.fasta target_2.fasta"  # Space-separated FASTA files
PARAMS="test_specific_query.csv"
TRAINING_FILE="HPBC_user_merged_num.csv"  # The training feature matrix
MODEL_NAME="HPBC"
TRAINING_SCORES="HPBC_target.csv"  # Training target labels/scores
PREDICTION_MODE="specific_query"  # Analysis mode
SPECIFIC_QUERY_INPUT="test_specific_query.csv"  # Custom query parameters for specific_query mode

# Record the start time
start_time=$(date)

# Specific query prediction mode
echo "Running in specific_query prediction mode..."
output_dir=$(pwd)  # Use the current directory where the script is run
python3 "$script_path" \
  --target_files_prediction $TARGET_FILES \
  --params $PARAMS \
  --prediction_mode $PREDICTION_MODE \
  --training_file $TRAINING_FILE \
  --training_scores $TRAINING_SCORES \
  --specific_query_input $SPECIFIC_QUERY_INPUT \
  --model_name $MODEL_NAME \
  --output_dir "$output_dir"

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
