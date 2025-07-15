import pandas as pd

# Load your CSV file
csv_path = "HPBC_default_ML_train.csv"  # change this if needed
target_column = "Y"  # update if your target column has a different name

# Read the CSV
df = pd.read_csv(csv_path)

# Check if target column exists
if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found in the dataset.")

# Count values
value_counts = df[target_column].value_counts()

# Display the counts
print("Target value counts:")
print(value_counts)

# Check if 0s and 1s are equal
if 0 in value_counts and 1 in value_counts:
    if value_counts[0] == value_counts[1]:
        print("✅ The number of 0s and 1s in the target column is equal.")
    else:
        print("❌ The number of 0s and 1s in the target column is NOT equal.")
else:
    print("⚠️ The target column does not contain both 0 and 1.")