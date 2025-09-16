#!/usr/bin/env python3
import pandas as pd
import sys

# Input and output file arguments
balanced_file = sys.argv[1] if len(sys.argv) > 1 else "HPBC_balanced_classification.csv"
train_file = sys.argv[2] if len(sys.argv) > 2 else "HPBC_ML_train.csv"
output_file = sys.argv[3] if len(sys.argv) > 3 else "filtered_train.csv"

# Load CSVs
balanced_df = pd.read_csv(balanced_file)
train_df = pd.read_csv(train_file)

# Drop duplicate id2 in train_file (keep first match)
train_unique = train_df.drop_duplicates(subset=["id2"], keep="first")

# Select only necessary columns from train
train_features = [
    "id2", "E_1", "Pu1_1", "Pu2_1", "E_hybrid_1",
    "seedNumber_1", "seedEbest_1", "E_3", "seedNumber_3",
    "E_diff_12", "pumin1_4u", "pumin5_8u", "pumin1_4d", "pumin5_8d"
]
train_selected = train_unique[train_features]

# Keep only id2 + ML_training_score (renamed to Y) from balanced file
balanced_subset = balanced_df[["id2", "ML_training_score"]].rename(columns={"ML_training_score": "Y"})

# Merge: keep all id2 from balanced, attach selected train features
result_df = balanced_subset.merge(train_selected, on="id2", how="left")

# Save result
result_df.to_csv(output_file, index=False)

print(f"Balanced has {len(balanced_subset)} rows. Output saved {len(result_df)} rows to {output_file}")
