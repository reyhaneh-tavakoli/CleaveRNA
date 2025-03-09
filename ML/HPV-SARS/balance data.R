# Load necessary libraries
library(dplyr)
library(tidyr)
library(readr)

# Define the threshold for categorizing Y
dataYthreshold <- 0.14

# Read and merge the two CSV files
mergedData <- bind_rows(
  read_csv("mergedData_annotated.num-HPV.csv"),
  read_csv("mergedData_annotated.num-SARS.csv")
)

# Categorize Y based on the threshold and convert to binary target
catData <- mergedData |>
  mutate(Y = if_else(Y >= dataYthreshold, 1, 0))

# Get mean and std of numeric data
catData |>
  select(-Y) |>
  summarise(across(everything(), list(mean = mean, sd = sd))) |>
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value") |>
  separate(feature, into = c("feature", "stat"), sep = "_(?=mean|sd)") |>
  pivot_wider(names_from = stat, values_from = value) |>
  write_csv("mergedData_annotated.mean.sd.csv")

# Get number of instances per class to have balanced results
nPerClass <- min(sum(catData$Y), nrow(catData) - sum(catData$Y))

# Define random seed for slice_sample
set.seed(89273554)

# Standardize all numeric data (excluding the Y column) and balance the dataset
catData |>
  mutate(across(-Y, \(col) scale(col, center = TRUE, scale = TRUE) |> as.vector())) |>
  group_by(Y) |>
  slice_sample(n = nPerClass) |>
  ungroup() |>
  write_csv("mergedData_annotated.cat.scaled.balanced.csv")