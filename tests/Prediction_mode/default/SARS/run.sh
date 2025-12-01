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
TARGET_FILES_PREDICTION="1.fasta 2.fasta 3.fasta 4.fasta 5.fasta 6.fasta 7.fasta 8.fasta 9.fasta 10.fasta 11.fasta 12.fasta 13.fasta 14.fasta 15.fasta 16.fasta 17.fasta 18.fasta 19.fasta 20.fasta"   # Space-separated FASTA files
PARAMS="test_default.csv"
TRAINING_FILE="SARS_user_merged_num.csv"  # The actual default train CSV file
MODEL_NAME="SARS"
TRAINING_SCORES="SARS_target.csv"

# Record the start time
start_time=$(date)

# prediction mode
echo "Running in prediction mode..."
output_dir=$(pwd)  # Use the current directory where the script is run
python3 "$script_path" \
  --target_files_prediction $TARGET_FILES_PREDICTION \
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

