#!/bin/bash

# Get the current directory (where the script is run)
CURRENT_DIR=$(pwd)

# Define script path
SCRIPT_PATH="$CURRENT_DIR/SVM-parameter.py"

# Define virtual environment directory
VENV_DIR="$CURRENT_DIR/ml_env"

# Log start time
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
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pandas numpy scikit-learn

# Run the Python script (no arguments needed)
python3 "$SCRIPT_PATH"

# Deactivate virtual environment
deactivate

# Log end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script finis

