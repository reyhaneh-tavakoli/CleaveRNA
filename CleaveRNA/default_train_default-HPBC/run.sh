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
script_dir="../CleaveRNA.py"  # Correctly reference CleaveRNA.py

# Define the input files and parameters
TARGETS="HPV.fasta BCL-1.fasta BCL-2.fasta BCL-3.fasta BCL-4.fasta BCL-5.fasta"  # Use space-separated file paths
PARAMS="test_default.csv"
DEFAULT_TRAIN_MODE="HPBC"
MODEL_NAME="HPBC"
ML_TARGET="HPBC_target.csv"

# Record the start time
start_time=$(date)

# Default mode
echo "Running in default mode..."
output_dir=$(pwd)  # Use the current directory where the script is run
python3 "$script_dir" \
  --targets $TARGETS \
  --params $PARAMS \
  --feature_mode default \
  --default_train_mode $DEFAULT_TRAIN_MODE \
  --model_name $MODEL_NAME \
  --ML_target $ML_TARGET \
  --output_dir "$output_dir"

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"

