import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import sys

# setting a static random number seed for ML 

# Load dataset
data_path = sys.argv[1]
output_dir = sys.argv[2]  # Directory to save results
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

df = pd.read_csv(data_path)

# Ensure target column 'Y' exists
target_column = 'Y'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Identify columns starting with 'pumin'
pumin_features = [col for col in X.columns if col.startswith("pumin")]

# Define hyperparameter grids
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

param_dist = {
    'C': np.logspace(-2, 2, 10),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Stratified 5-fold cross-validation
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
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
    
    return {
        "accuracy_mean": np.mean(accuracies), "accuracy_std": np.std(accuracies),
        "precision_mean": np.mean(precisions), "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls), "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s)
    }

# Perform hyperparameter tuning and evaluation
def tune_and_evaluate(X, y, param_grid, param_dist, skf):
    base_svm = SVC(probability=True)
    
    print("Starting Grid Search...")
    grid_search = GridSearchCV(base_svm, param_grid, cv=skf, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    best_svm_grid = grid_search.best_estimator_
    grid_results = evaluate_model(best_svm_grid, X, y, skf)
    grid_results['best_params'] = grid_search.best_params_
    print(f"Grid Search completed. Best Params: {grid_search.best_params_}")
    
    print("Starting Random Search...")
    random_search = RandomizedSearchCV(base_svm, param_dist, n_iter=10, cv=skf, scoring='f1', random_state=42, verbose=1, n_jobs=-1)
    random_search.fit(X, y)
    best_svm_random = random_search.best_estimator_
    random_results = evaluate_model(best_svm_random, X, y, skf)
    random_results['best_params'] = random_search.best_params_
    print(f"Random Search completed. Best Params: {random_search.best_params_}")
    
    return grid_results, random_results

# Evaluate on different feature sets
results = []

# Generate all feature combinations from 5 to 13 features
feature_list = list(X.columns)
for num_features in range(5, min(14, len(feature_list) + 1)):
    for i, feature_subset in enumerate(combinations(feature_list, num_features)):
        print(f"Running feature combination {i+1} for {num_features} features...")
        X_subset = X[list(feature_subset)]
        grid_res, random_res = tune_and_evaluate(X_subset, y, param_grid, param_dist, skf)
        results.append([f"{num_features} Features: {feature_subset}", "Grid Search", *grid_res.values()])
        results.append([f"{num_features} Features: {feature_subset}", "Random Search", *random_res.values()])
        print(f"Completed feature combination {i+1} for {num_features} features.")

# Save results
output_path = os.path.join(output_dir, "svm_hyperparameter_results.csv")
pd.DataFrame(results, columns=[
    "Feature Set", "Search Method", "Accuracy Mean", "Accuracy Std", "Precision Mean", "Precision Std", 
    "Recall Mean", "Recall Std", "F1 Mean", "F1 Std", "Best Parameters"
]).to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
