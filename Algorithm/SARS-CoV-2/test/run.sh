#!/bin/bash

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Define directories using absolute paths
script_dir="/home/reyhaneh/Documents/git/RNAcutter/RNAcutter.py"
output_dir="./output"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# activate environment with dependencies
conda activate intarna_env

# Record the start time
start_time=$(date)

# Process all fasta files in the current directory
for fasta_file in *.fasta; do
    # Skip temp files that start with "temp_"
    if [[ $fasta_file == temp_* ]] || [[ $fasta_file == output_* ]]; then
        continue
    fi
    
    # Check if file exists and is regular file
    if [ -f "$fasta_file" ]; then
        # Get the base name of the fasta file
        base_name=$(basename "$fasta_file" .fasta)
        
        echo "Processing file: $fasta_file"
        
        # Run the RNAcutter.py script
        python "${script_dir}" -motifs=AU,GU,AC,GC -LA=16 -RA=7 -target="$fasta_file"  
        
        # Check if the processing was successful
        if [ $? -eq 0 ]; then
            # Rename and move the output CSV files to the output directory
            mv "Results_with_region.csv" "$output_dir/${base_name}_output1.csv"
            mv "Results_without_region.csv" "$output_dir/${base_name}_output2.csv"
            mv "Results_pairwise.csv" "$output_dir/${base_name}_output3.csv"
            
            # Rename temp files if they exist
            mv "temp_sequence.fasta" "$output_dir/${base_name}_sequence.fasta" 2>/dev/null || true
            
            # Rename output_queries.fasta if it exists
            mv "output_queries.fasta" "$output_dir/${base_name}_queries.fasta" 2>/dev/null || true
            
            # Rename all temp_query files
            for temp_query in temp_query_*.fasta; do
                if [ -f "$temp_query" ]; then
                    query_num=$(echo "$temp_query" | sed 's/temp_query_\(.*\)\.fasta/\1/')
                    mv "$temp_query" "$output_dir/${base_name}_query_${query_num}.fasta" 2>/dev/null || true
                fi
            done
            
            echo "Successfully processed: $fasta_file"
        else
            echo "Error processing: $fasta_file"
        fi
        
        echo "------------------------"
    fi
done

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
