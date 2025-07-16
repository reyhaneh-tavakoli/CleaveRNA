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

# Define the specific feature set you want to use
selected_features = [
    'Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1',
    'E_3', 'seedNumber_3', 'pumin1_4d', 'pumin5_8d'
]

# Ensure all features exist
missing_features = [f for f in selected_features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

# Prepare X and y
X = df[selected_features]
y = df[target_column]

# Define SVM parameter grid
param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"]
}
param_combinations = list(itertools.product(param_grid["kernel"], param_grid["C"], param_grid["gamma"]))

# Stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to evaluate model
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

# Evaluate for each hyperparameter combination
results = []
for kernel, C, gamma in param_combinations:
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    result = evaluate_model(model, X, y, skf)

    results.append([
        f"SVM (kernel={kernel}, C={C}, gamma={gamma})",
        ", ".join(selected_features),
        kernel, C, gamma,
        result["accuracy_mean"], result["accuracy_std"],
        result["precision_mean"], result["precision_std"],
        result["recall_mean"], result["recall_std"],
        result["f1_mean"], result["f1_std"]
    ])

# Convert to DataFrame and save
results_df = pd.DataFrame(results, columns=[
    "Model", "Feature Set", "Kernel", "C", "Gamma",
    "Accuracy Mean", "Accuracy Std",
    "Precision Mean", "Precision Std",
    "Recall Mean", "Recall Std",
    "F1 Mean", "F1 Std"
])

results_df = results_df.sort_values(by="F1 Mean", ascending=False)

output_path = os.path.join(output_dir, "svm_selected_feature_results.csv")
results_df.to_csv(output_path, index=False)

print(f"✅ SVM results for selected feature set saved to: {output_path}")