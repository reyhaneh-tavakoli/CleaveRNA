#!/bin/bash

# Define directories
script_dir="../../"
output_dir="./output"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# activate environment with dependencies
# conda activate IntaRNA


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
    python $(script_dir)/RNAcutter.py -motifs=AU,GU -LA=9 -RA=16 -target="$fasta_file"  
    
    # Rename and move the output CSV file to the output directory
    mv "RNAcutter_result.csv" "$output_dir/${base_name}_output.csv"

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
