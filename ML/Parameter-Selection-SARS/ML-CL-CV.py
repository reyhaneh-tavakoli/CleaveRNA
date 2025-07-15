import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import itertools

# Load dataset directly
data_path = "SARS_default_ML_train.csv"
output_dir = "ML_output"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)

# Ensure target column 'Y' exists
target_column = 'Y'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_column, 'id2']) if 'id2' in df.columns else df.drop(columns=[target_column])
y = df[target_column]

# Define models to evaluate
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True)
}

# Stratified 5-fold cross-validation
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

# Run combinations of all possible feature sets with 8 to 13 features from all columns
all_feature_columns = X.columns
for r in range(8, min(13, len(all_feature_columns)) + 1):
    for feature_subset in itertools.combinations(all_feature_columns, r):
        X_subset = X[list(feature_subset)]
        feature_name = ", ".join(feature_subset)
        for model_name, model in models.items():
            result = evaluate_model(model, X_subset, y, skf)
            results.append([model_name, feature_name, *result.values()])

# Save results in the specified output directory, sorted by F1 Mean (descending)
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

print(f"Results saved to {output_path}")