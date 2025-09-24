# Record the start time
start_time=$(date)
# Ensure virtual environment is active
if [ ! -d "myenv" ]; then
  python3 -m venv myenv
fi
source myenv/bin/activate

# Update pip to the latest version
python3 -m pip install --upgrade pip

# Install necessary libraries if not already installed
pip install pandas numpy seaborn matplotlib shap plotly argparse

# Navigate to the directory where ML.py is located
cd ~/Documents/git/RNAcutter/ML

# Run the Python script
python3 ML-CV.py ~/Documents/git/RNAcutter/ML/SARS-CoV-2/mergedData_annotated.num.csv

 # Create MLResult directory if it doesn't exist
echo "Creating MLResult directory"
mkdir -p MLResult
# Move all generated files to MLResult
echo "Moving output files to MLResult directory..."
mv *.csv *.png MLResult/

# Check if the plots folder exists and move it to the HPV directory
if [ -d "MLResult" ]; then
  mv MLResult ~/Documents/git/RNAcutter/ML/SARS-CoV-2/
  echo "MLResult moved to ~/Documents/git/RNAcutter/ML/SARS-CoV-2"
else
  echo "No MLResult folder found!"
fi

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
