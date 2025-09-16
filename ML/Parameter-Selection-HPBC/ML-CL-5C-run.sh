#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Define paths
SCRIPT_PATH="$CURRENT_DIR/ML-CL-CV.py"  # Replace with your actual script filename
INPUT_CSV="$CURRENT_DIR/HPBC_ML_train.csv"
OUTPUT_DIR="$CURRENT_DIR/ML_output"

# Log start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $START_TIME"
echo "Current directory: $CURRENT_DIR"

# Check if input CSV exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input file $INPUT_CSV not found!"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script $SCRIPT_PATH not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script with PyMOL's packages in the path
echo "Running Python script..."
PYTHONPATH="/home/reytakop/Documents/software/Pymol/pymol/lib/python3.10/site-packages:$PYTHONPATH" \
python3 "$SCRIPT_PATH"

# Check if script ran successfully
if [ $? -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "ERROR: Python script failed to run."
    exit 1
fi

# Log end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script finished at: $END_TIME"
