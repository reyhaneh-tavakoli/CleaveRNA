# Load necessary libraries
library(dplyr)
library(tidyr)
library(readr)

# Define the threshold for categorizing Y
dataYthreshold <- 0.14

# Read and merge the two CSV files
mergedData <- bind_rows(
  read_csv("mergedData_annotated.num-HPV.csv"),
  read_csv("mergedData_annotated.num-SARS.csv"),
  read_csv("mergedData_annotated.num-BCL.csv")
)

# Step 1: Categorize Y into three categories
catData <- mergedData |>
  mutate(Category = case_when(
    Y == 0 ~ 0,                  # First category: Y == 0
    Y > 0 & Y < dataYthreshold ~ 1,  # Second category: 0 < Y < 0.14
    Y >= dataYthreshold ~ 2      # Third category: Y >= 0.14
  )) |>
  select(-Y)  # Remove original Y column after categorization

# Step 2: Balance the dataset across categories
nPerClass <- min(table(catData$Category))  # Find smallest category size

set.seed(89273554)  # Ensure reproducibility

balancedData <- catData |>
  group_by(Category) |>
  slice_sample(n = nPerClass) |>
  ungroup()

# Step 3: Standardize all numeric columns (excluding Category)
standardizedData <- balancedData |>
  mutate(across(-Category, \(col) scale(col, center = TRUE, scale = TRUE) |> as.vector()))

# Save the processed dataset
write_csv(standardizedData, "mergedData_annotated.cat.scaled.balanced.csv")
