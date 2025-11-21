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
script_dir="../../CleaveRNA.py"  # Path to CleaveRNA.py
PREDICTION_TRAIN_MODE="1.fasta 2.fasta 3.fasta 4.fasta 5.fasta 6.fasta 7.fasta 8.fasta 9.fasta 10.fasta 11.fasta 12.fasta 13.fasta 14.fasta 15.fasta 16.fasta 17.fasta 18.fasta 19.fasta 20.fasta"   # Space-separated FASTA files for training
PARAMS="test_default.csv"


# Record the start time
start_time=$(date)

echo "Running in train mode..."
output_dir=$(pwd)
python3 "$script_dir" --training_mode $PREDICTION_TRAIN_MODE --params $PARAMS --feature_mode default --output_dir "$output_dir" --model_name "SARS" 

# Record the end time
end_time=$(date)

echo "Start time: $start_time"
echo "End time: $end_time"
