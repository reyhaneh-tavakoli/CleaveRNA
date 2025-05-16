#!/bin/bash

# Define input CSV file
INPUT_CSV="./HPV-SARS-BCL/mergedData_annotated.cat.scaled.balanced.csv"

# Define script path
SCRIPT_PATH="$HOME/Documents/git/RNAcutter/ML/ML-CL-CV.py"

# Get start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $START_TIME"

# Run Python script
python3 "$SCRIPT_PATH" "$INPUT_CSV"

# Get end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script finished at: $END_TIME"

