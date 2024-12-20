#!/bin/bash

# Define directories
input_dir="/home/reyhaneh/Documents/git/RNAcutter/Algorithm/SARS-CoV-2 data/input"
script_dir="/home/reyhaneh/Documents/git/RNAcutter/Algorithm/SARS-CoV-2 data/script"
output_dir="/home/reyhaneh/Documents/git/RNAcutter/Algorithm/SARS-CoV-2 data/output"
#python_path="/home/reyhaneh/.asdf/installs/python/3.11.0/bin/python"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Record the start time
start_time=$(date)

# Loop through each .fasta file in the input directory
for fasta_file in "$input_dir"/*.fasta; do
    # Get the base name of the fasta file
    base_name=$(basename "$fasta_file" .fasta)
    
    # Print the name of the fasta file
    echo "Processing file: $fasta_file"
    
    # Run the RNAcutter.py script
    python $(script_dir)/RNAcutter.py -motifs=AU,GU -LA=9 -RA=16 -target="$fasta_file"
    
    # Rename and move the output CSV file to the output directory
    mv "RNAcutter_result.csv" "$output_dir/${base_name}_output.csv"
done

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
