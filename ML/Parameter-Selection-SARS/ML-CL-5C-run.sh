#!/bin/bash

# Get the current directory (where the script is run)
CURRENT_DIR=$(pwd)

# Define paths
SCRIPT_PATH="$CURRENT_DIR/ML-CL-CV.py"
INPUT_CSV="$CURRENT_DIR/SARS_default_ML_train.csv"
OUTPUT_DIR="$CURRENT_DIR/results"
VENV_DIR="$CURRENT_DIR/ml_env"

# Log start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $START_TIME"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip and install required packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pandas numpy scikit-learn shap

# Run Python script with input CSV and output directory
python3 "$SCRIPT_PATH" "$INPUT_CSV" "$OUTPUT_DIR"

# Deactivate virtual environment
deactivate

# Log end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script finished at: $END_TIME"

