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

USER_TRAIN_MODE="HPV.fasta BCL-1.fasta BCL-2.fasta"  # Space-separated FASTA files for training
TARGETS="test.fasta"  # Space-separated FASTA files for prediction
PARAMS="test_default.csv"
ML_TARGET="MERS_target.csv"  # Path to your target CSV file

# Record the start time
start_time=$(date)

echo "Running in user train mode..."
output_dir=$(pwd)
python3 "$script_dir" --user_train_mode $USER_TRAIN_MODE --targets $TARGETS --params $PARAMS --feature_mode default --output_dir "$output_dir" --model_name "MERS" --ML_target "$ML_TARGET"

# Record the end time
end_time=$(date)

echo "Start time: $start_time"
echo "End time: $end_time"
