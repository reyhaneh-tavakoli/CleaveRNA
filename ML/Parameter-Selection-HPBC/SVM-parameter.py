import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import itertools

# Load dataset
data_path = "HPBC_default_ML_train.csv"
output_dir = "ML_output"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)

# Ensure target column exists
target_column = 'Y'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_column, 'id2']) if 'id2' in df.columns else df.drop(columns=[target_column])
y = df[target_column]

# SVM hyperparameter grid
param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"]
}
param_combinations = list(itertools.product(param_grid["kernel"], param_grid["C"], param_grid["gamma"]))

# Stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare result storage
results = []

def evaluate_model(model, X, y, skf):
    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    return {
        "accuracy_mean": np.mean(accuracies), "accuracy_std": np.std(accuracies),
        "precision_mean": np.mean(precisions), "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls), "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s)
    }

# Loop over feature combinations (8 to 13 features)
all_features = list(X.columns)
min_features = min(13, len(all_features))
for r in range(8, min_features + 1):
    for feature_subset in itertools.combinations(all_features, r):
        X_subset = X[list(feature_subset)]
        feature_name = ", ".join(feature_subset)

        for kernel, C, gamma in param_combinations:
            model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
            result = evaluate_model(model, X_subset, y, skf)

            results.append([
                f"SVM (kernel={kernel}, C={C}, gamma={gamma})",
                feature_name,
                kernel, C, gamma,
                result["accuracy_mean"], result["accuracy_std"],
                result["precision_mean"], result["precision_std"],
                result["recall_mean"], result["recall_std"],
                result["f1_mean"], result["f1_std"]
            ])

# Convert results to DataFrame and sort
results_df = pd.DataFrame(results, columns=[
    "Model", "Feature Set", "Kernel", "C", "Gamma",
    "Accuracy Mean", "Accuracy Std",
    "Precision Mean", "Precision Std",
    "Recall Mean", "Recall Std",
    "F1 Mean", "F1 Std"
])

results_df = results_df.sort_values(by="F1 Mean", ascending=False)

# Save to CSV
output_path = os.path.join(output_dir, "svm_feature_param_results.csv")
results_df.to_csv(output_path, index=False)

print(f"✅ All combinations tested. Results saved to: {output_path}")