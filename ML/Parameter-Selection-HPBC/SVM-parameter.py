import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Load dataset
data_path = "HPBC_ML_train.csv"
output_dir = "ML_output"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)

# Ensure target column exists
target_column = 'Y'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# Define specific feature set
selected_features = ['E_1', 'Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 'seedNumber_3', 'pumin1_4u', 'pumin1_4d']

# Check if all selected features exist in the dataset
missing_features = [f for f in selected_features if f not in df.columns]
if missing_features:
    print(f"Warning: Missing features in dataset: {missing_features}")
    print(f"Available features: {list(df.columns)}")
    # Use only available features
    selected_features = [f for f in selected_features if f in df.columns]
    print(f"Using available features: {selected_features}")

# Separate features and target
X = df[selected_features]
y = df[target_column]

print(f"Using feature set: {selected_features}")
print(f"Dataset shape: {X.shape}")

# Extended SVM hyperparameter grid for thorough exploration
param_grid = {
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "C": [0.01, 0.1, 1, 10, 100, 1000],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10],
    "degree": [2, 3, 4, 5],  # Only for poly kernel
    "coef0": [0.0, 0.1, 1.0]  # For poly and sigmoid kernels
}

# Generate parameter combinations based on kernel type
param_combinations = []

# Linear kernel - only C matters
for C in param_grid["C"]:
    param_combinations.append(("linear", C, "scale", 3, 0.0))

# RBF kernel - C and gamma matter
for C in param_grid["C"]:
    for gamma in param_grid["gamma"]:
        param_combinations.append(("rbf", C, gamma, 3, 0.0))

# Polynomial kernel - C, gamma, degree, and coef0 matter
for C in param_grid["C"]:
    for gamma in param_grid["gamma"]:
        for degree in param_grid["degree"]:
            for coef0 in param_grid["coef0"]:
                param_combinations.append(("poly", C, gamma, degree, coef0))

# Sigmoid kernel - C, gamma, and coef0 matter
for C in param_grid["C"]:
    for gamma in param_grid["gamma"]:
        for coef0 in param_grid["coef0"]:
            param_combinations.append(("sigmoid", C, gamma, 3, coef0))

print(f"Total parameter combinations to test: {len(param_combinations)}")

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

# Loop over parameter combinations for the selected feature set
feature_name = ", ".join(selected_features)

for i, (kernel, C, gamma, degree, coef0) in enumerate(param_combinations):
    print(f"Testing combination {i+1}/{len(param_combinations)}: kernel={kernel}, C={C}, gamma={gamma}, degree={degree}, coef0={coef0}")
    
    # Create SVM model with appropriate parameters based on kernel
    if kernel == "linear":
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    elif kernel == "rbf":
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    elif kernel == "poly":
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0, probability=True, random_state=42)
    elif kernel == "sigmoid":
        model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0, probability=True, random_state=42)
    
    try:
        result = evaluate_model(model, X, y, skf)
        
        results.append([
            f"SVM (kernel={kernel}, C={C}, gamma={gamma}, degree={degree}, coef0={coef0})",
            feature_name,
            kernel, C, gamma, degree, coef0,
            result["accuracy_mean"], result["accuracy_std"],
            result["precision_mean"], result["precision_std"],
            result["recall_mean"], result["recall_std"],
            result["f1_mean"], result["f1_std"]
        ])
    except Exception as e:
        print(f"Error with parameters {kernel}, C={C}, gamma={gamma}: {e}")
        continue

# Convert results to DataFrame and sort
results_df = pd.DataFrame(results, columns=[
    "Model", "Feature Set", "Kernel", "C", "Gamma", "Degree", "Coef0",
    "Accuracy Mean", "Accuracy Std",
    "Precision Mean", "Precision Std",
    "Recall Mean", "Recall Std",
    "F1 Mean", "F1 Std"
])

results_df = results_df.sort_values(by="F1 Mean", ascending=False)

# Save to CSV
output_path = os.path.join(output_dir, "svm_specific_features_param_results.csv")
results_df.to_csv(output_path, index=False)

# Display top 10 results
print("\n" + "="*80)
print("TOP 10 BEST PERFORMING PARAMETER COMBINATIONS:")
print("="*80)
print(results_df.head(10).to_string(index=False))

print(f"\n✅ All parameter combinations tested. Full results saved to: {output_path}")
print(f"📊 Total combinations tested: {len(results_df)}")
print(f"🎯 Best F1 Score: {results_df['F1 Mean'].max():.4f}")
print(f"🏆 Best model: {results_df.iloc[0]['Model']}")