import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import seaborn as sns
import argparse

def save_metrics(metrics_dict, filename):
    """Save evaluation metrics to CSV file"""
    df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R² Score"],
        "Value": [metrics_dict["mae"], metrics_dict["mse"], 
                 metrics_dict["rmse"], metrics_dict["r2"]]
    })
    df.to_csv(filename, index=False)

def compute_metrics(y_true, y_pred):
    """Compute all evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred)
    }

def main(input_file):
    # Read and prepare data
    churn_df = pd.read_csv(input_file)
    churn_df2 = pd.get_dummies(churn_df, drop_first=True)
    
    # Split features and target
    y = churn_df2["Y"]
    X = churn_df2.drop("Y", axis=1)
    
    # Initial train-test split model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    model = RandomForestRegressor(random_state=10)
    fit = model.fit(X_train, y_train)
    yhat = fit.predict(X_test)
    
    # Save initial metrics
    metrics = compute_metrics(y_test, yhat)
    save_metrics(metrics, "data_scaled.csv")
    
    # Initial SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, features=X.columns, show=False)
    plt.savefig('shap_summary_plot1.png')
    plt.close()
    
    # Save initial SHAP values
    pd.DataFrame(shap_values, columns=X.columns).to_csv('shap_values_1.csv', index=False)
    
    # 5-fold Cross Validation
    CV = KFold(n_splits=5, shuffle=True, random_state=10)
    SHAP_values_per_fold = []
    new_index = []
    
    for i, (train_ix, test_ix) in enumerate(CV.split(churn_df2)):
        print(f'\n------ Fold Number: {i}')
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        
        model = RandomForestRegressor(random_state=10)
        fit = model.fit(X_train, y_train)
        yhat = fit.predict(X_test)
        
        metrics = compute_metrics(y_test, yhat)
        save_metrics(metrics, f"model_evaluation_fold_{i}.csv")
    print(churn_df[['E_1_BoxCox', 'Y']].corr())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze churn data using Random Forest and SHAP.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()
    main(args.input_file)