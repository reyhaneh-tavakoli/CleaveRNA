import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import subprocess
import glob

def debug_print(*args, **kwargs):
    """Helper function for debug printing"""
    print(*args, **kwargs)

def create_sars_config(output_file):
    config_content = """
mode=M
model=X
energy=V
temperature=23
acc=C
accW=150
accL=100
seedBP=5
outSep=,"""
    with open(output_file, 'w') as f:
        f.write(config_content)
    print(f"Created {output_file} with SARS-CoV-2 parameters")

def create_sars_txt(output_file):
    txt_content = """motifs=AU,GU
LA=16
RA=7"""
    with open(output_file, 'w') as f:
        f.write(txt_content)
    print(f"Created {output_file} with SARS-CoV-2 motifs and parameters")

def create_hpv_config(output_file):
    config_content = """
mode=M
model=X
energy=V
temperature=37
acc=C
accW=150
accL=100
seedBP=5
outSep=,"""
    with open(output_file, 'w') as f:
        f.write(config_content)
    print(f"Created {output_file} with HPV-BCL parameters")

def create_hpv_txt(output_file):
    txt_content = """motifs=AU,GU
LA=16
RA=7"""
    with open(output_file, 'w') as f:
        f.write(txt_content)
    print(f"Created {output_file} with HPV-BCL motifs and parameters")

def save_feature_stats(X_df, model_name):
    """Save mean and standard deviation of each feature to a CSV based on predefined values"""
    if model_name == 'SARS-CoV-2':
        # Predefined SARS-CoV-2 statistics
        stats_data = {
            'feature': [
                'E_1', 'Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 
                'E_3', 'seedNumber_3', 'E_diff_12', 'pumin1_4u', 'pumin5_8u', 
                'pumin1_4d', 'pumin5_8d'
            ],
            'mean': [
                -16.8143612334802, 0.00386839063132315, 0.0123355011509028, -28.5197797356828,
                8.37444933920705, -4.90176211453745, -8.85867841409692, 3.05286343612335,
                -8.15850220264317, 0.361770028269163, 0.271348412784141, 0.260072055105727,
                0.239206449225551
            ],
            'std': [
                3.70360012748828, 0.00876173817856928, 0.0667426085527343, 4.76311486588401,
                2.5858462661661, 1.72014340721644, 4.41140898343531, 1.56477491209297, 
                2.23320822710615, 0.272953540745652, 0.241853110779549, 0.232690244042691,
                0.226298031582116
            ]
        }
        stats_file = "SARS-train-statistic.csv"
        
    elif model_name == 'HPV-BCL':
        # Predefined HPV-BCL statistics
        stats_data = {
            'feature': [
                'E_1', 'Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 
                'E_3', 'seedNumber_3', 'E_diff_12', 'pumin1_4u', 'pumin5_8u', 
                'pumin1_4d', 'pumin5_8d'
            ],
            'mean': [
                -14.9912711864407, 0.00416939757682438, 0.0683360511442184, -25.1444915254237,
                7.80508474576271, -5.0956779661017, -8.54194915254237, 3.27118644067797,
                -0.0572033898305085, 0.244907686813559, 0.265481861025424, 0.240881206220339,
                0.208485827635593
            ],
            'std': [
                3.59178777155817, 0.0119028541130887, 0.171175470569825, 5.42164270016972,
                2.18486821983751, 1.80017163417876, 3.73569894339507, 1.40000620846702,
                0.274372276867028, 0.206971999132851, 0.219986115934916, 0.215514376007841,
                0.179081018969295
            ]
        }
        stats_file = "HPV-BCL-train-statistic.csv"
    
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Create DataFrame from the predefined values
    stats_df = pd.DataFrame(stats_data)
    
    # Save to CSV
    stats_df.to_csv(stats_file, index=False)
    print(f"Predefined feature statistics saved to {stats_file}")
    
    # Also save with standard filename format for compatibility
    standard_stats_file = f"{model_name}-feature-stats.csv"
    stats_df.to_csv(standard_stats_file, index=False)
    print(f"Also saved feature statistics to {standard_stats_file} for compatibility")

def train_and_save_svm(train_data_path, model_name):
    print(f"\nTraining SVM model for {model_name} using data from {train_data_path}")
    
    df = pd.read_csv(train_data_path)
    if 'Y' not in df.columns:
        raise ValueError("Training data must include a target column named 'Y'.")

    X_df = df.drop(columns=['Y'])
    y = df['Y']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X_df)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train SVM
    svm = SVC(C=10, gamma='auto', kernel='rbf', probability=True, random_state=42)
    svm.fit(X, y)

    # Save model
    model_file = f"{model_name}-SVM.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': svm,
            'scaler': scaler,
            'imputer': imputer,
            'feature_columns': X_df.columns.tolist()
        }, f)
    print(f"Saved trained model to {model_file}")

    # Cross-validation
    perform_cross_validation(X, y, model_name)

    # Save predefined feature stats
    save_feature_stats(X_df, model_name)

def perform_cross_validation(X, y, model_name):
    print(f"\nRunning 5-fold cross-validation for {model_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(C=10, gamma='auto', kernel='rbf', probability=True, random_state=42)

    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)

        scores['accuracy'].append(accuracy_score(y_test, preds))
        scores['precision'].append(precision_score(y_test, preds))
        scores['recall'].append(recall_score(y_test, preds))
        scores['f1'].append(f1_score(y_test, preds))

    print("\nCross-validation results:")
    for metric, values in scores.items():
        print(f"{metric.capitalize()}: {np.mean(values):.3f} ± {np.std(values):.3f}")

def post_process_features(target_file, output_dir):
    """Perform additional feature processing after Feature.py completes"""
    debug_print("\nStarting post-processing...")
    
    try:
        # Get the base name of the target file
        base_filename = os.path.splitext(os.path.basename(target_file))[0]
        
        # Define the RNAplfold output directory
        rnaplfold_dir = f"rnaplfold_output_{base_filename}"
        
        # Define the path to the _lunp file
        pu_file = os.path.join(rnaplfold_dir, f"{base_filename}_lunp")
        if not os.path.exists(pu_file):
            raise FileNotFoundError(f"✗ Unpaired probability file {pu_file} not found. Did RNAplfold run correctly?")
        
        print(f"✓ Reading unpaired probabilities from {pu_file}...")
        pu = pd.read_csv(pu_file, sep="\t", skiprows=2, header=None)
        pu.columns = ["i"] + [f"l{i}" for i in range(1, len(pu.columns))]
        debug_print("Unpaired data sample:\n", pu.head())
        
        # Verify we have position data
        if len(pu) == 0:
            raise ValueError("✗ Unpaired probability file is empty")

        # Read and process all three output files
        out = []
        result_files = {
            1: f"{base_filename}_Results_with_region.csv",
            2: f"{base_filename}_Results_without_region.csv",
            3: f"{base_filename}_Results_pairwise.csv"
        }
        
        for i, result_file in result_files.items():
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"✗ Result file {result_file} not found. Did IntaRNA run correctly?")
            
            print(f"✓ Processing {result_file}...")
            df = pd.read_csv(result_file)
            debug_print(f"Result file {i} sample:\n", df.head())
            
            # Verify required columns exist
            required_cols = ['id2', 'seq2', 'seedE']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"✗ Missing required columns in {result_file}: {missing_cols}")
            
            # Keep only first solution per sequence pair
            df = df.groupby('id2').first().reset_index()
            
            # Number of seeds
            df['seedNumber'] = df['seedE'].str.count(':') + 1
            df['seedEbest'] = df['seedE'].str.extract(r'^([^:]+)').astype(float)
            
            # Add suffix to column names
            df.columns = [f"{col}_{i}" for col in df.columns]
            out.append(df)

        # Merge datasets
        debug_print("Merging datasets...")
        merged_data = out[0].merge(out[1], 
                                 left_on=["id2_1", "seq2_1"],
                                 right_on=["id2_2", "seq2_2"],
                                 how='outer')
        merged_data = merged_data.merge(out[2], 
                                      left_on=["id2_1", "seq2_1"],
                                      right_on=["id2_3", "seq2_3"],
                                      how='outer')
        merged_data = merged_data.rename(columns={'id2_1': 'id2', 'seq2_1': 'seq2'})
        debug_print("Merged data columns:", merged_data.columns.tolist())
        debug_print("Merged data sample:\n", merged_data.head())

        # Fill NA values
        na_defaults = {
            'seedEbest': 0, 'seedNumber': 0, 'E': 0, 'Pu1': 0, 'Pu2': 0,
            'E_hybrid': 0, 'P_E': 0, 'ED1': 999999, 'ED2': 999999,
            'seedE': "0", 'seedPu1': "0", 'seedPu2': "0",
            'seedED1': "999999", 'seedED2': "999999"
        }

        for suffix in ['_1', '_2', '_3']:
            for col, default in na_defaults.items():
                full_col = col + suffix
                if full_col in merged_data.columns:
                    merged_data[full_col] = merged_data[full_col].fillna(default)

        # Calculate energy differences
        if 'E_1' in merged_data.columns:
            merged_data = merged_data[merged_data['E_1'] != 0]
            if 'E_2' in merged_data.columns:
                merged_data['E_diff_12'] = merged_data['E_2'] - merged_data['E_1']

        # Calculate unpaired probabilities
        if 'id2' in merged_data.columns:
            merged_data['pos'] = merged_data['id2'].str.extract(r'(\d+)').astype(float)
            
            def safe_get_min_pu(row, positions, direction):
                try:
                    pos = int(row['pos'])
                    if direction == 'u':
                        return min(pu.loc[pos - p, 'l1'] for p in positions)
                    else:
                        return min(pu.loc[pos + 1 + p, 'l1'] for p in positions)
                except (ValueError, KeyError):
                    return float('nan')
            
            for name, positions, direction in [
                ('pumin1_4u', [1,2,3,4], 'u'),
                ('pumin5_8u', [5,6,7,8], 'u'),
                ('pumin1_4d', [1,2,3,4], 'd'),
                ('pumin5_8d', [5,6,7,8], 'd')
            ]:
                merged_data[name] = merged_data.apply(
                    lambda x: safe_get_min_pu(x, positions, direction), axis=1)
            
            merged_data = merged_data.drop(columns=['pos'])

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full merged data with file identifier
        full_output = os.path.join(output_dir, f"{base_filename}_mergedData_full.csv")
        merged_data.to_csv(full_output, index=False)
        print(f"✓ Saved full merged data to {full_output}")
        debug_print("Full data sample:\n", merged_data.head())

        # Create feature sets
        feature_sets = {
            'feature_set_1': ['E_1', 'Pu1_1', 'E_hybrid_1', 'seedNumber_1', 
                            'seedEbest_1', 'seedNumber_3', 'E_diff_12',
                            'pumin1_4u', 'pumin1_4d', 'pumin5_8d'],
            'feature_set_2': ['E_1', 'Pu1_1', 'Pu2_1', 'E_hybrid_1',
                           'seedEbest_1', 'seedNumber_3', 'pumin1_4u',
                           'pumin1_4d', 'pumin5_8d']
        }

        for feature_name, columns in feature_sets.items():
            available_cols = [col for col in columns if col in merged_data.columns]
            if available_cols:
                output_path = os.path.join(output_dir, f"{base_filename}_{feature_name}.csv")
                merged_data[available_cols].to_csv(output_path, index=False)
                print(f"✓ Saved {feature_name} to {output_path}")
            else:
                print(f"⚠ Warning: No available columns for {feature_name}")

        return True

    except Exception as e:
        print(f"✗ Error during post-processing: {str(e)}")
        return False

def run_feature_mode(args, target_file, output_dir):
    """Run the feature generation mode using the config file for a single target file"""
    print(f"\n=== RUNNING FEATURE MODE FOR {os.path.basename(target_file)} ===")
    feature_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Feature.py")
    
    # Verify target file exists
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"✗ Target file {target_file} not found")
    
    # Build the command
    config_file = 'SARS-parameter.cfg' if args.model == 'SARS-CoV-2' else 'HPV-parameter.cfg'

    # Extract parameters from the generated config file
    with open(config_file, 'r') as f:
        config_params = dict(line.strip().split('=') for line in f if '=' in line)

    # Extract parameters from the generated txt file
    txt_file = 'SARS-parameter.txt' if args.model == 'SARS-CoV-2' else 'HPV-parameter.txt'
    with open(txt_file, 'r') as f:
        txt_params = dict(line.strip().split('=') for line in f if '=' in line)

    cmd = [
        "python3", feature_script,
        f"--cfg={os.path.abspath(config_file)}",
        f"--target={os.path.abspath(target_file)}",
        f"--motifs={txt_params.get('motifs', 'AU,GU')}",
        f"--LA={txt_params.get('LA', '16')}",
        f"--RA={txt_params.get('RA', '7')}"
    ]
    
    try:
        # Run with full path visibility
        print("Running Feature.py...")
        debug_print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        
        # Post-processing after Feature.py completes
        print("\nRunning post-processing...")
        return post_process_features(target_file, output_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Feature generation failed with error: {e}")
        debug_print("Error output:", e.stderr)
        return False

def manual_standardize(data, feature_stats):
    """Apply manual standardization using feature means and stds from the training statistics file"""
    standardized_data = data.copy()
    
    for feature in data.columns:
        if feature in feature_stats.index:
            mean = feature_stats.at[feature, 'mean']
            std = feature_stats.at[feature, 'std']
            if pd.notna(std) and std > 0:  # Avoid division by zero or NaN
                standardized_data[feature] = (data[feature] - mean) / std
            else:
                standardized_data[feature] = 0  # Set to zero if std is zero or NaN
    
    return standardized_data

def run_predict_mode(model_name, output_dir):
    print("\n=== RUNNING PREDICT MODE ===")

    # Check if output directory exists
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} not found.")

    # Find all merged data files
    merged_csv_pattern = os.path.join(output_dir, "*_mergedData_full.csv")
    merged_csv_files = glob.glob(merged_csv_pattern)
    
    if not merged_csv_files:
        raise FileNotFoundError(f"No merged data files found in {output_dir}. Run feature mode first.")
    
    # Load model
    model_file = f"{model_name}-SVM.pkl"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found. Please train the model first.")
    
    with open(model_file, 'rb') as f:
        model_bundle = pickle.load(f)

    model = model_bundle['model']
    imputer = model_bundle['imputer']
    feature_columns = model_bundle['feature_columns']
    
    # Load feature statistics from the model-specific file
    if (model_name == 'SARS-CoV-2'):
        stats_file = "SARS-train-statistic.csv"
    else:  # HPV-BCL
        stats_file = "HPV-BCL-train-statistic.csv"
    
    if not os.path.exists(stats_file):
        # Fallback to standard filename format
        stats_file = f"{model_name}-feature-stats.csv"
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"Neither model-specific statistics file nor {stats_file} found. Please train the model first.")
    
    print(f"Loading feature statistics from {stats_file}")
    feature_stats = pd.read_csv(stats_file)
    # Convert to a format for easy lookup
    feature_stats = feature_stats.set_index('feature')
    
    # Process each merged data file
    for merged_csv_path in merged_csv_files:
        base_filename = os.path.basename(merged_csv_path).replace("_mergedData_full.csv", "")
        print(f"\nProcessing predictions for {base_filename}...")
        
        # Load merged data to get id2 and seq2
        merged_df = pd.read_csv(merged_csv_path)
        id_seq_df = merged_df[['id2', 'seq2']].copy()
        
        # Find feature set files for this base filename
        feature_set_paths = {
            'set1': os.path.join(output_dir, f"{base_filename}_feature_set_1.csv"),
            'set2': os.path.join(output_dir, f"{base_filename}_feature_set_2.csv")
        }
        
        stat_paths = {
            'set1': os.path.join(output_dir, f"{base_filename}_feature_set_1_stats.csv"),
            'set2': os.path.join(output_dir, f"{base_filename}_feature_set_2_stats.csv")
        }
        
        for set_key, set_path in feature_set_paths.items():
            print(f"\nProcessing {set_key.upper()} for {base_filename}...")

            if not os.path.exists(set_path):
                print(f"⚠ {set_path} not found. Skipping.")
                continue

            # Load feature set
            df = pd.read_csv(set_path)
            print("Expected feature columns:", feature_columns)
            print("Available columns in feature set:", df.columns.tolist())

            # Filter to use only columns that are both in the model and in the current data
            available_columns = [col for col in feature_columns if col in df.columns]
            if not available_columns:
                print(f"⚠ No matching feature columns found in {set_path}. Skipping.")
                continue

            print(f"Using {len(available_columns)} available features: {available_columns}")
            
            # Create a DataFrame with just the available columns from the feature set
            X_available = df[available_columns].copy()
            
            # Create a new DataFrame with all expected columns initialized to 0
            X_full = pd.DataFrame(0, index=range(len(X_available)), columns=feature_columns)
            
            # Fill in the available columns with actual data
            for col in available_columns:
                X_full[col] = X_available[col]
            
            # Save stats of current data
            current_stats_df = pd.DataFrame({
                'feature': X_available.columns,
                'mean': X_available.mean(),
                'std': X_available.std()
            })
            current_stats_df.to_csv(stat_paths[set_key], index=False)
            print(f"✓ Saved current feature stats to {stat_paths[set_key]}")

            # Apply imputation first
            X_imputed = imputer.transform(X_full)
            X_imputed_df = pd.DataFrame(X_imputed, columns=feature_columns)
            
            # Apply manual standardization using saved feature statistics
            X_std_df = manual_standardize(X_imputed_df, feature_stats)
            
            # Save standardized features
            standardized_csv = os.path.join(output_dir, f"{base_filename}_{set_key}_Standardized.csv")
            X_std_df.to_csv(standardized_csv, index=False)
            print(f"✓ Saved standardized features to {standardized_csv}")

            # Convert to numpy array for model
            X_std = X_std_df.values

            # Predict
            y_pred = model.predict(X_std)
            
            # Add probability scores if available
            try:
                y_proba = model.predict_proba(X_std)
                result_df = id_seq_df.copy()
                result_df['Predicted_Y'] = y_pred
                if y_proba.shape[1] >= 2:  # Binary classification
                    result_df['Probability'] = y_proba[:, 1]  # Probability of class 1
                else:
                    result_df['Probability'] = y_proba[:, 0]
            except:
                # If predict_proba not available or fails
                result_df = id_seq_df.copy()
                result_df['Predicted_Y'] = y_pred

            # Save prediction result
            out_file = os.path.join(output_dir, f"{base_filename}_{set_key}_predictions.csv")
            result_df.to_csv(out_file, index=False)
            print(f"✓ Prediction result saved to {out_file}")

def combine_results(output_dir):
    """Combine prediction results from multiple files into one summary file"""
    print("\n=== COMBINING PREDICTION RESULTS ===")
    
    # Find all prediction files
    prediction_files = glob.glob(os.path.join(output_dir, "*_set1_predictions.csv"))
    
    if not prediction_files:
        print("⚠ No prediction files found to combine.")
        return
    
    all_results = []
    
    for pred_file in prediction_files:
        base_filename = os.path.basename(pred_file).replace("_set1_predictions.csv", "")
        df = pd.read_csv(pred_file)
        df['Source_File'] = base_filename
        all_results.append(df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(output_dir, "combined_predictions.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"✓ Combined {len(all_results)} prediction results into {combined_file}")
        
        # Generate summary statistics
        positive_count = (combined_df['Predicted_Y'] == 1).sum()
        total_count = len(combined_df)
        pos_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\nSummary Statistics:")
        print(f"Total predictions: {total_count}")
        print(f"Positive predictions: {positive_count} ({pos_percentage:.2f}%)")
        print(f"Negative predictions: {total_count - positive_count} ({100 - pos_percentage:.2f}%)")
        
        # Save summary by source file
        if 'Source_File' in combined_df.columns:
            summary = combined_df.groupby('Source_File').agg(
                Total=('Predicted_Y', 'count'),
                Positive=('Predicted_Y', lambda x: (x == 1).sum()),
                Negative=('Predicted_Y', lambda x: (x == 0).sum())
            ).reset_index()
            
            summary['Positive_Percent'] = (summary['Positive'] / summary['Total']) * 100
            
            summary_file = os.path.join(output_dir, "prediction_summary_by_file.csv")
            summary.to_csv(summary_file, index=False)
            print(f"✓ Saved prediction summary by file to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Run RNA pipeline: training + feature + prediction')
    parser.add_argument('targets', nargs='+', help='One or more target FASTA files (used in feature & prediction)')
    parser.add_argument('--model', required=True, choices=['SARS-CoV-2', 'HPV-BCL'], help='Model type')
    parser.add_argument('--train_data', help='Path to training data CSV (required for training)')
    parser.add_argument('--train', action='store_true', help='Run training only')
    parser.add_argument('--feature', action='store_true', help='Run feature generation only')
    parser.add_argument('--predict', action='store_true', help='Run prediction only')
    parser.add_argument('--run_all', action='store_true', help='Run full pipeline: training + feature + prediction')
    parser.add_argument('--output_dir', default='feature_outputs', help='Directory for output files')

    args = parser.parse_args()

    # Set up config and txt files
    if args.model == 'SARS-CoV-2':
        config_file = 'SARS-parameter.cfg'
        create_sars_config(config_file)
        create_sars_txt('SARS-parameter.txt')
    else:
        config_file = 'HPV-parameter.cfg'
        create_hpv_config(config_file)
        create_hpv_txt('HPV-parameter.txt')

    print(f"\nTarget FASTA file(s): {args.targets}")
    print(f"Model: {args.model}")
    print(f"Parameter config written to: {config_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_all:
        if not args.train_data:
            print("✗ Error: --run_all requires --train_data")
            return

        print("\n=== RUNNING IN FULL PIPELINE MODE (TRAIN + FEATURE + PREDICT) ===")
        # Training phase
        train_and_save_svm(args.train_data, args.model)
        
        # Process each target file
        all_success = True
        for target_file in args.targets:
            # Create a subdirectory for each target file in the output directory
            target_base_name = os.path.splitext(os.path.basename(target_file))[0]
            target_output_dir = os.path.join(args.output_dir, target_base_name)
            os.makedirs(target_output_dir, exist_ok=True)

            # Run feature mode for the current target file
            success = run_feature_mode(args, target_file, target_output_dir)
            if not success:
                print(f"✗ Feature mode failed for {target_file}. Skipping this file.")
                all_success = False

        # Run prediction if any features were successfully generated
        if all_success:
            for target_file in args.targets:
                target_base_name = os.path.splitext(os.path.basename(target_file))[0]
                target_output_dir = os.path.join(args.output_dir, target_base_name)
                run_predict_mode(args.model, target_output_dir)
        else:
            print("✗ Some feature generations failed. Check logs for details.")
    else:
        if args.train:
            if not args.train_data:
                print("✗ Error: --train requires --train_data")
                return
            train_and_save_svm(args.train_data, args.model)

        if args.feature:
            for target_file in args.targets:
                run_feature_mode(args, target_file, args.output_dir)

        if args.predict:
            run_predict_mode(args.model, args.output_dir)
            combine_results(args.output_dir)  # Combine results from multiple files

if __name__ == "__main__":
    main()