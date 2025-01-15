#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Define directories
script_dir="../../RNAcutter.py"
output_dir="./output"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# activate environment with dependencies
conda activate intarna_env

# Record the start time
start_time=$(date)

# input file
fasta_file=sequence.fasta

# Get the base name of the fasta file
base_name=$(basename "$fasta_file" .fasta)

# Print the name of the fasta file
echo "Processing file: $fasta_file"

# Run the RNAcutter.py script
# TODO UPDATE PARAMETERS FOR HPV
python "${script_dir}" -motifs=AU,GU,AC,GC -LA=9 -RA=8 -target="$fasta_file"  

# Rename and move the output CSV files to the output directory
mv "Results_with_region.csv" "$output_dir/${base_name}_output1.csv"
mv "Results_without_region.csv" "$output_dir/${base_name}_output2.csv"
mv "Results_pairwise.csv" "$output_dir/${base_name}_output3.csv"

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
