import pandas as pd
import os
import numpy as np

def analyze_top_parameters(data_path, file_pattern_base, dataset_names, output_filename):
    """
    Analyze top 50 F1 scores in Model columns and find parameters with high F1 means in both files.
    
    Parameters:
    data_path (str): Path to the directory containing CSV files
    file_pattern_base (str): Base pattern for file names (e.g., "comparition_results")
    dataset_names (list): List of dataset names (e.g., ["HPBC", "SARS"])
    output_filename (str): Name of the output CSV file
    """
    
    # Dictionary to store individual dataset dataframes
    dataset_dfs = {}
    
    # Read CSV files for each dataset
    for dataset in dataset_names:
        file_path = os.path.join(data_path, f"{file_pattern_base}.{dataset}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataset_dfs[dataset] = df
            print(f"Successfully read {file_path}")
        else:
            print(f"Warning: File {file_path} not found")
    
    if not dataset_dfs:
        print("No CSV files found. Please check the file paths.")
        return
    
    # Check for required columns
    for dataset, df in dataset_dfs.items():
        if 'Model' not in df.columns or 'F1 Mean' not in df.columns:
            print(f"Required columns 'Model' and 'F1 Mean' not found in {dataset} data")
            print(f"Available columns in {dataset}:", df.columns.tolist())
            return
    
    # Analyze top 50 F1 scores in each dataset separately
    top_50_per_dataset = {}
    
    for dataset, df in dataset_dfs.items():
        # Sort by F1 Mean in descending order and get top 50
        top_50_df = df.nlargest(50, 'F1 Mean')[['Model', 'F1 Mean']].copy()
        top_50_df = top_50_df.rename(columns={'F1 Mean': f'F1_Mean_{dataset}'})
        top_50_per_dataset[dataset] = top_50_df
        print(f"\nTop 5 models in {dataset}:")
        print(top_50_df.head())
    
    # Merge all datasets to find common high-performing parameters
    all_dfs = []
    for dataset, df in dataset_dfs.items():
        df_copy = df[['Model', 'F1 Mean']].copy()
        df_copy['dataset'] = dataset
        all_dfs.append(df_copy)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Pivot to have datasets as columns
    pivoted_df = combined_df.pivot_table(
        index='Model', 
        columns='dataset', 
        values='F1 Mean', 
        aggfunc='first'
    ).reset_index()
    
    # Fill NaN values with 0 (in case some models are missing in one dataset)
    pivoted_df = pivoted_df.fillna(0)
    
    # Calculate statistics for each model across datasets
    dataset_cols = [col for col in pivoted_df.columns if col != 'Model']
    pivoted_df['mean_F1'] = pivoted_df[dataset_cols].mean(axis=1)
    pivoted_df['min_F1'] = pivoted_df[dataset_cols].min(axis=1)
    pivoted_df['max_F1'] = pivoted_df[dataset_cols].max(axis=1)
    pivoted_df['std_F1'] = pivoted_df[dataset_cols].std(axis=1)
    pivoted_df['jointF1'] = pivoted_df[dataset_cols].sum(axis=1)
    
    # Find parameters that perform well in BOTH datasets (high minimum F1)
    high_performers_both = pivoted_df[pivoted_df['min_F1'] > 0.7].copy()  # Adjust threshold as needed
    
    # Sort by mean F1 score in descending order
    sorted_df = pivoted_df.sort_values('mean_F1', ascending=False)
    
    # Select top 50 by mean F1 score
    top_50_mean = sorted_df.head(50)
    
    # Sort by joint F1 score for comparison
    sorted_joint = pivoted_df.sort_values('jointF1', ascending=False)
    top_50_joint = sorted_joint.head(50)
    
    # Create comprehensive analysis results
    analysis_results = {
        'top_50_by_mean_F1': top_50_mean,
        'top_50_by_joint_F1': top_50_joint,
        'high_performers_both_datasets': high_performers_both.sort_values('mean_F1', ascending=False),
        'individual_dataset_top_50': top_50_per_dataset
    }
    
    # Generate only the high_performers_both.csv file
    if not high_performers_both.empty:
        # Sort by mean F1 score in descending order (highest to lowest F1)
        high_performers_both = high_performers_both.sort_values(['mean_F1', 'min_F1'], ascending=[False, False])
        
        output_path_both = os.path.join(data_path, "high_performers_both.csv")
        high_performers_both.to_csv(output_path_both, index=False)
        print(f"\nHigh performers in both datasets saved to: {output_path_both}")
        print(f"Found {len(high_performers_both)} models performing well in both datasets")
        print("Results sorted from highest to lowest F1 scores")
        
        print("\nTop 10 parameters performing well in both datasets (sorted highest to lowest F1):")
        print(high_performers_both[['Model', 'mean_F1', 'min_F1'] + dataset_cols].head(10))
    else:
        print("\nNo parameters found with high F1 scores in both datasets (threshold: 0.7)")
        print("Lowering threshold to 0.5 to find some common performers...")
        
        # Try with lower threshold
        high_performers_both_lower = pivoted_df[pivoted_df['min_F1'] > 0.5].copy()
        if not high_performers_both_lower.empty:
            # Sort by mean F1 score in descending order (highest to lowest F1)
            high_performers_both_lower = high_performers_both_lower.sort_values(['mean_F1', 'min_F1'], ascending=[False, False])
            
            output_path_both = os.path.join(data_path, "high_performers_both.csv")
            high_performers_both_lower.to_csv(output_path_both, index=False)
            print(f"High performers (threshold 0.5) saved to: {output_path_both}")
            print(f"Found {len(high_performers_both_lower)} models performing well in both datasets")
            print("Results sorted from highest to lowest F1 scores")
            
            print("\nTop 10 parameters performing well in both datasets (threshold 0.5, sorted highest to lowest F1):")
            print(high_performers_both_lower[['Model', 'mean_F1', 'min_F1'] + dataset_cols].head(10))
        else:
            print("No parameters found even with lower threshold (0.5)")
            # If no models meet the threshold, save top 50 by mean F1 score
            top_50_by_mean = pivoted_df.sort_values(['mean_F1', 'min_F1'], ascending=[False, False]).head(50)
            output_path_both = os.path.join(data_path, "high_performers_both.csv")
            top_50_by_mean.to_csv(output_path_both, index=False)
            print(f"Top 50 models by mean F1 saved to: {output_path_both}")
            print("Results sorted from highest to lowest F1 scores")
            print("\nTop 10 models by mean F1 score:")
            print(top_50_by_mean[['Model', 'mean_F1', 'min_F1'] + dataset_cols].head(10))
    
    return analysis_results

def main():
    # Configuration
    data_path = "/home/reytakop/Documents/Documents-Galaxy-server/git/CleaveRNA/ML/SVM-parameter"
    file_pattern_base = "svm_specific_features_param_results"
    dataset_names = ["HPBC", "SARS"]
    output_filename = "SVM-parameterHPBC-SARS.top50.csv"
    
    # Run the analysis process
    result = analyze_top_parameters(data_path, file_pattern_base, dataset_names, output_filename)
    
    if result is not None:
        print("\nProcessing completed successfully!")
        print("Generated file: high_performers_both.csv - Models performing well in both datasets")

if __name__ == "__main__":
    main()