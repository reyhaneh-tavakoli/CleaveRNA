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
    save_metrics(metrics, "model_evaluation1.csv")
    
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
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        SHAP_values_per_fold.extend(shap_values)
        new_index.extend(test_ix)
    
    # Save CV SHAP plot
    plt.figure()
    shap.summary_plot(np.array(SHAP_values_per_fold), X.iloc[new_index], show=False)
    plt.savefig('shap_summary_plot2.png')
    plt.close()
    
    # Multiple CV repeats analysis
    np.random.seed(1)
    CV_repeats = 10
    random_states = np.random.randint(10000, size=CV_repeats)
    
    shap_values_per_cv = {sample: {CV_repeat: {} 
                                  for CV_repeat in range(CV_repeats)} 
                         for sample in X.index}
    
    for cv_repeat in range(CV_repeats):
        print(f'\n------------ CV Repeat number: {cv_repeat}')
        CV = KFold(n_splits=5, shuffle=True, random_state=random_states[cv_repeat])
        
        for fold_num, (train_ix, test_ix) in enumerate(CV.split(churn_df2)):
            print(f'\n------ Fold Number: {fold_num}')
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
            
            model = RandomForestRegressor(random_state=10)
            fit = model.fit(X_train, y_train)
            yhat = fit.predict(X_test)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            for i, test_index in enumerate(test_ix):
                shap_values_per_cv[test_index][cv_repeat] = shap_values[i]
    
    # Calculate final statistics
    average_shap_values = []
    stds = []
    ranges = []
    
    for i in range(len(churn_df2)):
        df_per_obs = pd.DataFrame.from_dict(shap_values_per_cv[i])
        average_shap_values.append(df_per_obs.mean(axis=1).values)
        stds.append(df_per_obs.std(axis=1).values)
        ranges.append(df_per_obs.max(axis=1).values - df_per_obs.min(axis=1).values)
    
    # Save final SHAP plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(np.array(average_shap_values), X, show=False)
    plt.title('Average SHAP values after 10x cross-validation')
    plt.savefig('shap_summary_plot_final.png')
    plt.close()
    
    # Save standard deviations
    pd.DataFrame(stds, columns=X.columns).to_csv('shap_values_stds.csv', index=False)
    
    # Create and save range plots
    ranges_df = pd.DataFrame(ranges, columns=X.columns)
    long_df = pd.melt(ranges_df, var_name='Features', value_name='Values')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long_df, x='Features', y='Values')
    plt.xticks(rotation=45)
    plt.title('Range of SHAP values per sample across all\ncross-validation repeats')
    plt.xlabel('SHAP Value Variability')
    plt.ylabel('SHAP range per sample')
    plt.tight_layout()
    plt.savefig('shap_ranges_plot.png')
    plt.close()
    
    # Create and save standardized range plot
    standardized = long_df.copy()
    standardized['Values'] = standardized.groupby('Features')['Values'].transform(lambda x: x/x.mean())
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=standardized, x='Features', y='Values')
    plt.xticks(rotation=45)
    plt.title('Scaled Range of SHAP values per sample\nacross all cross-validation repeats')
    plt.xlabel('SHAP Value Variability Scaled by Mean')
    plt.ylabel('Scaled SHAP range')
    plt.tight_layout()
    plt.savefig('shap_ranges_standardized_plot.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze churn data using Random Forest and SHAP.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()
    main(args.input_file)