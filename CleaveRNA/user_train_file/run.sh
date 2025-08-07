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
script_dir="../CleaveRNA.py"  # Path to CleaveRNA.py
PREDICTION_TRAIN_MODE="MERS.fasta"  # Space-separated FASTA files for training
PARAMS="test_default.csv"


# Record the start time
start_time=$(date)

echo "Running in train mode..."
output_dir=$(pwd)
python3 "$script_dir" --train_mode $PREDICTION_TRAIN_MODE --params $PARAMS --feature_mode default --output_dir "$output_dir" --model_name "MERS" 

# Record the end time
end_time=$(date)

echo "Start time: $start_time"
echo "End time: $end_time"
