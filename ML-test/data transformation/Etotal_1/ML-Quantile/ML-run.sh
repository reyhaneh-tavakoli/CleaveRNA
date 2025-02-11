# Record the start time
start_time=$(date)
# Ensure virtual environment is active
if [ ! -d "myenv" ]; then
  python3 -m venv myenv
fi
source myenv/bin/activate

# Update pip to the latest version
python3 -m pip install --upgrade pip

pip install scikit-learn --break-system-packages
# Install necessary libraries if not already installed
pip install xgboost pandas numpy seaborn matplotlib shap plotly argparse


# Run the Python script
python3 ML-CV.py data_Qtransformed.csv

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
