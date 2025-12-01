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

# Define script and input files
script_dir="../../../CleaveRNA/CleaveRNA.py"  # Path to CleaveRNA.py
TARGET_FILES_TRAINING="HPV.fasta BCL-1.fasta BCL-2.fasta BCL-3.fasta BCL-4.fasta BCL-5.fasta"   # Space-separated FASTA files for training
PARAMS="test_default.csv"


# Record the start time
start_time=$(date)

echo "Running in train mode..."
output_dir=$(pwd)
python3 "$script_dir" --target_files_training $TARGET_FILES_TRAINING --params $PARAMS --prediction_mode default --output_dir "$output_dir" --model_name "HPBC" 

# Record the end time
end_time=$(date)

echo "Start time: $start_time"
echo "End time: $end_time"
