import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import itertools
import datetime

# -----------------------------
# Setup
# -----------------------------
data_path = "HPBC_ML_train.csv"
output_dir = "ML_output"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "run_report.log")

# Helper function to log messages
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(data_path)
target_column = 'Y'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

X = df.drop(columns=[target_column, 'id2']) if 'id2' in df.columns else df.drop(columns=[target_column])
y = df[target_column]

# -----------------------------
# Define models
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# -----------------------------
# Evaluation function
# -----------------------------
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

# -----------------------------
# Run feature combinations
# -----------------------------
all_feature_columns = X.columns
total_combinations = sum(1 for r in range(8, 14) for _ in itertools.combinations(all_feature_columns, r))
log(f"Total feature combinations to evaluate: {total_combinations}")

count = 0
for r in range(8, 14):
    for feature_subset in itertools.combinations(all_feature_columns, r):
        count += 1
        feature_name = ", ".join(feature_subset)
        log(f"START combination {count}/{total_combinations}: {feature_name}")
        combo_start_time = datetime.datetime.now()

        X_subset = X[list(feature_subset)]
        for model_name, model in models.items():
            model_start_time = datetime.datetime.now()
            log(f"  START model {model_name} for combination {count}")
            result = evaluate_model(model, X_subset, y, skf)
            results.append([model_name, feature_name, *result.values()])
            model_end_time = datetime.datetime.now()
            log(f"  END model {model_name} for combination {count} | Duration: {model_end_time - model_start_time}")

        combo_end_time = datetime.datetime.now()
        log(f"END combination {count}/{total_combinations}: {feature_name} | Duration: {combo_end_time - combo_start_time}")

# -----------------------------
# Save results
# -----------------------------
output_path = os.path.join(output_dir, "comparition_results.csv")
results_df = pd.DataFrame(results, columns=[
    "Model", "Feature Set",
    "Accuracy Mean", "Accuracy Std",
    "Precision Mean", "Precision Std",
    "Recall Mean", "Recall Std",
    "F1 Mean", "F1 Std"
])
results_df = results_df.sort_values(by="F1 Mean", ascending=False)
results_df.to_csv(output_path, index=False)
log(f"All results saved to {output_path}")
