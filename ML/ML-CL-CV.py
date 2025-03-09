import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shap
import sys

# Load dataset
data_path = sys.argv[1]
df = pd.read_csv(data_path)

# Ensure target column 'Y' exists
target_column = 'Y'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_column])
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
shap_results = []

def evaluate_model(model, X, y, skf):
    accuracies, precisions, recalls, f1s = [], [], [], []
    shap_values = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        
        shap_values.append(np.mean(abs(explainer.shap_values(X_test, check_additivity=False)), axis=0))
    
    return {
        "accuracy_mean": np.mean(accuracies), "accuracy_std": np.std(accuracies),
        "precision_mean": np.mean(precisions), "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls), "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "shap_values": np.mean(shap_values, axis=0)
    }

full_feature_results = {}
for model_name, model in models.items():
    result = evaluate_model(model, X, y, skf)
    full_feature_results[model_name] = result
    results.append([model_name, "All Features", *result.values()])
    shap_results.append([model_name, "All Features", *result["shap_values"]])

for feature in X.columns:
    X_reduced = X.drop(columns=[feature])
    for model_name, model in models.items():
        result = evaluate_model(model, X_reduced, y, skf)
        results.append([model_name, f"Without {feature}", *result.values()])
        shap_results.append([model_name, f"Without {feature}", *result["shap_values"]])

output_path = "./HPV-SARS-BCL/comparition_results.csv"
pd.DataFrame(results, columns=["Model", "Feature Set", "Accuracy Mean", "Accuracy Std", "Precision Mean", "Precision Std", "Recall Mean", "Recall Std", "F1 Mean", "F1 Std", "SHAP Values"]).to_csv(output_path, index=False)

shap_output_path = "./HPV-SARS-BCL/SHAP_results.csv"
pd.DataFrame(shap_results, columns=["Model", "Feature Set"] + list(X.columns)).to_csv(shap_output_path, index=False)

print(f"Results saved to {output_path}")
print(f"SHAP feature importance saved to {shap_output_path}")
