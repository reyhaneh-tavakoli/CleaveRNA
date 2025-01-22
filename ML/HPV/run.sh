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
pip install pandas numpy seaborn matplotlib plotly ipython scipy Jinja2

# Navigate to the directory where ML.py is located
cd ~/Documents/git/RNAcutter/ML

# Run the Python script
python3 importance.py ~/Documents/git/RNAcutter/ML/HPV/mergedData_annotated.num.csv

# Check if the plots folder exists and move it to the HPV directory
if [ -d "plots" ]; then
  mv plots ~/Documents/git/RNAcutter/ML/HPV/
  echo "Plots moved to ~/Documents/git/RNAcutter/ML/HPV"
else
  echo "No plots folder found!"
fi

# Record the end time
end_time=$(date)

# Print the start and end times
echo "Start time: $start_time"
echo "End time: $end_time"
