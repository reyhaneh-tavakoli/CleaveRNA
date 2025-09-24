import pandas as pd

# Set the threshold
dataYthreshold = 0.14

# Read the CSV file
merged_data = pd.read_csv("mergedData_annotated.num.csv")

# Categorize Y based on the threshold
cat_data = merged_data.copy()
cat_data['Y'] = (cat_data['Y'] >= dataYthreshold).astype(int)

# Optional: Save the result to a new CSV file
cat_data.to_csv("mergedData_annotated.categorized.csv", index=False)

# Print a preview
print(cat_data.head())
