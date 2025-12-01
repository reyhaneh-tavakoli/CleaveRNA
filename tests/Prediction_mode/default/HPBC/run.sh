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
TARGET_FILES="HPV.fasta BCL-1.fasta BCL-2.fasta BCL-3.fasta BCL-4.fasta BCL-5.fasta" # Space-separated FASTA files
PARAMS="test_default.csv"
TRAINING_FILE="HPBC_user_merged_num.csv"  # The training feature matrix file
MODEL_NAME="HPBC"
TRAINING_SCORES="HPBC_target.csv"  # The training target labels/scores file

# Record the start time
start_time=$(date)

# prediction mode
echo "Running in prediction mode..."
output_dir=$(pwd)  # Use the current directory where the script is run
python3 "$script_path" \
  --target_files_prediction $TARGET_FILES \
  --params $PARAMS \
  --prediction_mode default \
  --training_file $TRAINING_FILE \
  --model_name $MODEL_NAME \
  --training_scores $TRAINING_SCORES \
  --output_dir "$output_dir"

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"

