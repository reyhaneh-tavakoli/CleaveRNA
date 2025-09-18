#!/bin/bash

# -----------------------------
# Setup
# -----------------------------
PYTHON_SCRIPT="ML-CL-CV.py"
OUTPUT_DIR="ML_output"
LOG_FILE="$OUTPUT_DIR/run_report.log"

mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "===============================" | tee "$LOG_FILE"
echo "ML run started at: $START_TIME" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"

# Step 1: Verify Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script $PYTHON_SCRIPT not found!" | tee -a "$LOG_FILE"
    exit 1
fi
echo "Step 1: Python script found." | tee -a "$LOG_FILE"

# Step 2: Create output directory
echo "Step 2: Output directory $OUTPUT_DIR is ready." | tee -a "$LOG_FILE"

# Step 3: Start ML evaluation
echo "Step 3: Starting ML evaluation..." | tee -a "$LOG_FILE"
python3 "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

# Step 4: End of ML evaluation
echo "Step 4: ML evaluation completed." | tee -a "$LOG_FILE"

# Record end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "===============================" | tee -a "$LOG_FILE"
echo "ML run ended at: $END_TIME" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"

# Step 5: Completion message
echo "Step 5: All combinations evaluated." | tee -a "$LOG_FILE"
echo "Check output CSV: $OUTPUT_DIR/comparition_results.csv" | tee -a "$LOG_FILE"
echo "Check detailed log: $LOG_FILE" | tee -a "$LOG_FILE"

