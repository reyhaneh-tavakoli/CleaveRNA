#!/bin/bash
pip3 install shap pandas numpy sklearn sys

# Get the current directory (where the script is run)
CURRENT_DIR=$(pwd)

# Define input CSV file (assumed to be inside the current directory)
INPUT_CSV="$CURRENT_DIR/mergedData_annotated.cat.scaled.balanced.csv"

# Define output directory (inside the current directory)
OUTPUT_DIR="$CURRENT_DIR/results"

# Define script path
SCRIPT_PATH="$HOME/Documents/git/RNAcutter/ML/ML-CL-CV.py"

# Define virtual environment path inside the current directory
VENV_DIR="$CURRENT_DIR/ml_env"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Get start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $START_TIME"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip and install required Python packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install shap

# Run Python script with input CSV and output directory
python3 "$SCRIPT_PATH" "$INPUT_CSV" "$OUTPUT_DIR"

# Deactivate virtual environment
deactivate

# Get end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script finished at: $END_TIME"

