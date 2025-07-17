import argparse
import os
import sys
import traceback
from Feature import main as feature_main
import pandas as pd
import numpy as np
from matplotlib import cm
import subprocess
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from scipy.stats import entropy
import pickle
import tempfile
import shutil

def create_cfg_file(params_file):
    """Generates the parameters.cfg file with settings based on the given params CSV file."""
    import pandas as pd

    # Read the temperature from the params CSV file
    params_df = pd.read_csv(params_file)
    if 'Tem' not in params_df.columns:
        raise ValueError("The 'Tem' column is missing in the parameters CSV file.")

    temperature = params_df['Tem'].iloc[0]  # Use the first value in the 'Tem' column

    cfg_content = f"""mode=M
model=X
energy=V
temperature={temperature}
acc=C
accW=150
accL=100
seedBP=5
outSep=,
"""
    # Save the parameters.cfg file in the same directory as the main script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file_path = os.path.join(script_dir, "parameters.cfg")
    with open(cfg_file_path, "w") as cfg_file:
        cfg_file.write(cfg_content)
    print(f"Configuration file 'parameters.cfg' created at {cfg_file_path} with temperature from params CSV.")

def report_file_status(file_path, description):
    if os.path.exists(file_path):
        print(f"Success: {description} generated successfully at {file_path}.")
    else:
        print(f"Error: {description} was not generated.")

def report_empty_file(file_path, description):
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == 0:
            print(f"Warning: {description} is empty and will be skipped: {file_path}.")
            return True
    else:
        print(f"Warning: {description} does not exist: {file_path}.")
        return True
    return False

def train_and_save_svm(train_data_path, model_name, feature_set_name):
    print(f"\nTraining SVM model for {model_name} using data from {train_data_path}")

    df = pd.read_csv(train_data_path)
    if 'Y' not in df.columns:
        raise ValueError("Training data must include a target column named 'Y'.")

    # Check class balance and balance if needed
    y_counts = df['Y'].value_counts()
    if len(y_counts) == 2 and y_counts.iloc[0] != y_counts.iloc[1]:
        print(f"Class imbalance detected: {y_counts.to_dict()}. Balancing by downsampling...")
        min_count = y_counts.min()
        df_balanced = df.groupby('Y', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
        df = df_balanced
        print(f"Balanced class counts: {df['Y'].value_counts().to_dict()}")
    elif len(y_counts) == 2:
        print(f"Classes are already balanced: {y_counts.to_dict()}")
    else:
        print(f"Warning: Only one class present in target column. SVM training may fail.")

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
    model_file = f"{model_name}-{feature_set_name}-SVM.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': svm,
            'scaler': scaler,
            'imputer': imputer,
            'feature_columns': X_df.columns.tolist()
        }, f)
    print(f"Saved trained model to {model_file}")

    # Cross-validation
    perform_cross_validation(X, y, model_name, feature_set_name)

def perform_cross_validation(X, y, model_name, feature_set_name):
    print(f"\nRunning 5-fold cross-validation for {model_name} ({feature_set_name})...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(C=10, gamma='auto', kernel='rbf', probability=True, random_state=42)

    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    proba_results = []  # Store predict_proba results for each fold

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)
        proba = svm.predict_proba(X_test)
        proba_results.append(proba)

        scores['accuracy'].append(accuracy_score(y_test, preds))
        scores['precision'].append(precision_score(y_test, preds))
        scores['recall'].append(recall_score(y_test, preds))
        scores['f1'].append(f1_score(y_test, preds))

    print("\nCross-validation results:")
    for metric, values in scores.items():
        print(f"{metric.capitalize()}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    # Optionally print or process the probability results
    print("\nSample predict_proba output from first fold:")
    if proba_results:
        print(proba_results[0])

    # Save cross-validation results to a CSV file
    metrics_file = f"{model_name}_ML_metrics_{feature_set_name}.csv"
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Mean': [np.mean(scores['accuracy']), np.mean(scores['precision']), np.mean(scores['recall']), np.mean(scores['f1'])],
        'Std': [np.std(scores['accuracy']), np.std(scores['precision']), np.std(scores['recall']), np.std(scores['f1'])]
    })
    metrics_df.to_csv(metrics_file, index=False)
    report_file_status(metrics_file, f"ML metrics for {feature_set_name}")

def train(args):
    # Ensure the output directory exists
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory set to: {args.output_dir}")

    # Ensure Feature.py is executed to generate the required file
    print("Running Feature.py to generate 'all_generated_merged_num.csv'...")
    # Convert the list of targets to a comma-separated string
    targets_arg = ','.join(args.targets)
    # Ensure the correct working directory is set for Feature.py
    feature_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Feature.py'))
    feature_command = f"python3 {feature_script_path} --targets {targets_arg} --params {args.params} --feature_mode {args.feature_mode}"
    try:
        subprocess.run(feature_command, shell=True, check=True, cwd=args.output_dir)
        print("Feature.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Feature.py execution failed with error: {e}")
        sys.exit(1)

    # Retry mechanism to check for 'all_generated_merged_num.csv'
    import time
    retries = 5
    output_file_path = os.path.join(args.output_dir, "all_generated_merged_num.csv")
    for attempt in range(retries):
        if os.path.exists(output_file_path):
            print(f"Success: 'all_generated_merged_num.csv' was generated at {output_file_path}.")
            break
        else:
            print(f"Attempt {attempt + 1}/{retries}: 'all_generated_merged_num.csv' not found at {output_file_path}. Retrying in 2 seconds...")
            time.sleep(2)
    else:
        print(f"Error: 'all_generated_merged_num.csv' was not generated by Feature.py after multiple attempts. Checked path: {output_file_path}")
        sys.exit(1)

    if args.default_train_file:
        model_name = args.default_train_file

        # File paths
        default_merged_file = f"{model_name}_default_merged_num.csv"
        target_file = f"{model_name}_target.csv"

        if not os.path.exists(default_merged_file) or not os.path.exists(target_file):
            print(f"Error: Required files '{default_merged_file}' or '{target_file}' do not exist.")
            sys.exit(1)

        # Check for empty result files
        result_files = [
            f"{model_name}_Results_with_region.csv",
            f"{model_name}_Results_without_region.csv",
            f"{model_name}_Results_pairwise.csv"
        ]

        for result_file in result_files:
            if report_empty_file(result_file, "Result file"):
                continue

        # Calculate mean and std for default_merged_file
        df_default = pd.read_csv(default_merged_file)
        mean_std = df_default.describe().loc[['mean', 'std']]
        mean_std_file = f"{model_name}_default_train_statistics.csv"
        mean_std.to_csv(mean_std_file)
        report_file_status(mean_std_file, "Default train statistics")

        # Standardize numerical columns while keeping the 'id2' column
        df_standardized = df_default.copy()
        for column in df_standardized.columns:
            if column != 'id2':
                df_standardized[column] = (df_standardized[column] - mean_std.loc['mean', column]) / mean_std.loc['std', column]
        standardized_file = f"{model_name}_standardized_default_train.csv"
        df_standardized.to_csv(standardized_file, index=False)
        report_file_status(standardized_file, "Standardized default train")

        # Balance target file
        df_target = pd.read_csv(target_file)
        np.random.seed(89273554)
        df_balanced = df_target.groupby('Y', group_keys=False).apply(
            lambda x: x.sample(df_target['Y'].value_counts().min(), random_state=89273554)
        )
        balanced_file = f"{model_name}_balanced_target.csv"
        df_balanced.to_csv(balanced_file, index=False)
        report_file_status(balanced_file, "Balanced target")

        df_non_balanced = df_target[~df_target.index.isin(df_balanced.index)]
        non_balanced_file = f"{model_name}_non_balanced_target.csv"
        df_non_balanced.to_csv(non_balanced_file, index=False)
        report_file_status(non_balanced_file, "Non-balanced target")

        # Merge standardized_default_train with balanced_target
        df_merged_train = pd.merge(df_standardized, df_balanced, on='id2')
        merged_train_file = f"{model_name}_default_ML_train.csv"
        df_merged_train.to_csv(merged_train_file, index=False)
        report_file_status(merged_train_file, "Default ML train")

        # Merge standardized_default_train with non_balanced_target
        df_merged_test = pd.merge(df_standardized, df_non_balanced, on='id2')
        merged_test_file = f"{model_name}_default_ML_test.csv"
        df_merged_test.to_csv(merged_test_file, index=False)
        report_file_status(merged_test_file, "Default ML test")

        # Define the single feature set
        feature_set = ['Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 'E_3', 'seedNumber_3', 'pumin1_4d', 'pumin5_8d']

        df_train = pd.read_csv(merged_train_file)
        feature_set_with_y = feature_set + ['Y']
        feature_set_file = f"{model_name}_default_ML_train_feature_set.csv"
        df_train[feature_set_with_y].to_csv(feature_set_file, index=False)
        report_file_status(feature_set_file, "Default ML train feature set")

        # Ensure proper standardization of columns in all_generated_merged_num.csv using HPBC_default_train_statistics.csv
        mean_std_file = f"{model_name}_default_train_statistics.csv"
        mean_std = pd.read_csv(mean_std_file, index_col=0)
        df_generated = pd.read_csv("all_generated_merged_num.csv")
        df_standardized_generated = df_generated.copy()

        # Standardize only columns present in mean_std
        for column in mean_std.columns:
            if column in df_standardized_generated.columns:
                df_standardized_generated[column] = (df_standardized_generated[column] - mean_std.loc['mean', column]) / mean_std.loc['std', column]

        # Retain id2 column and standardized columns
        df_standardized_generated = df_standardized_generated[['id2'] + list(mean_std.columns)]
        standardized_generated_file = "standardized_all_generated_merged_num.csv"
        df_standardized_generated.to_csv(standardized_generated_file, index=False)
        report_file_status(standardized_generated_file, "Standardized generated merged num")

        # Extract feature set from standardized_generated_merged_num
        generated_feature_set_file = "generated_ML_test_feature_set.csv"
        df_standardized_generated[feature_set].to_csv(generated_feature_set_file, index=False)
        report_file_status(generated_feature_set_file, "Generated ML test feature set")

        # Train and save SVM model for the single feature set
        train_and_save_svm(f"{model_name}_default_ML_train_feature_set.csv", model_name, "default_train_feature_set")

        # Predict for default train mode
        print("\nProcessing predictions for default_train_feature_set...")
        pickle_file = f"{model_name}-default_train_feature_set-SVM.pkl"
        test_file = "generated_ML_test_feature_set.csv"
        output_file = "feature_set_predicted.csv"
        model_file = os.path.join(args.output_dir, pickle_file)
        if not os.path.exists(model_file):
            print(f"\u26a0 {model_file} not found. Skipping.")
        else:
            with open(model_file, 'rb') as f:
                model_bundle = pickle.load(f)
            model = model_bundle['model']
            imputer = model_bundle['imputer']
            feature_columns = model_bundle['feature_columns']
            test_file_path = os.path.join(args.output_dir, test_file)
            if not os.path.exists(test_file_path):
                print(f"\u26a0 {test_file_path} not found. Skipping.")
            else:
                df_test = pd.read_csv(test_file_path)
                available_columns = [col for col in feature_columns if col in df_test.columns]
                if not available_columns:
                    print(f"\u26a0 No matching feature columns found in {test_file_path}. Skipping.")
                else:
                    X_full = pd.DataFrame(0, index=range(len(df_test)), columns=feature_columns)
                    for col in available_columns:
                        X_full[col] = df_test[col]
                    X_imputed = imputer.transform(X_full)
                    scaler = StandardScaler()
                    X_std = scaler.fit_transform(X_imputed)
                    y_true = None
                    if 'Y' in df_test.columns:
                        y_true = df_test['Y'].reset_index(drop=True)
                        if len(y_true) != len(X_full):
                            y_true = y_true.iloc[:len(X_full)].reset_index(drop=True)
                    y_pred, reliability_score, decision_score, entropies, _ = predict_with_confidence(model, X_std, y_true)
                    # Calculate predict_proba for predicted data
                    predict_proba = model.predict_proba(X_std)[:, 1] if model.predict_proba(X_std).shape[1] >= 2 else model.predict_proba(X_std)[:, 0]
                    margin = np.abs(decision_score)
                    combined_score = reliability_score * margin
                    result_df = pd.DataFrame({
                        'id2': df_test['id2'] if 'id2' in df_test.columns else range(len(df_test)),
                        'y_pred': y_pred,
                        'reliability_score': reliability_score,
                        'predict_proba': predict_proba,
                        'decision_score': decision_score,
                        'margin': margin,
                        'entropy': entropies,
                        'combined_score': combined_score
                    })
                    # Add target_file column after id2 using all_generated_merged_num.csv
                    all_gen = pd.read_csv(os.path.join(args.output_dir, "all_generated_merged_num.csv"))
                    id2_to_target = dict(zip(all_gen['id2'], all_gen['target_file'])) if 'target_file' in all_gen.columns else {}
                    # Insert target_file column after id2
                    result_df.insert(1, 'target_file', result_df['id2'].map(id2_to_target) if id2_to_target else None)
                    output_path = os.path.join(args.output_dir, f"{args.default_train_file}_{output_file}")
                    result_df.to_csv(output_path, index=False)
                    print(f"\u2713 Prediction result saved to {output_path}")

        # Save id2 and seq2 columns of all_generated_merged_num as CS_Dz.csv
        df_all_generated = pd.read_csv("all_generated_merged_num.csv")
        cs_dz_file = "CS_Dz.csv"
        # Add target_file column as well
        df_all_generated[['id2', 'seq2', 'target_file']].to_csv(cs_dz_file, index=False)
        report_file_status(cs_dz_file, "CS_Dz file")

        # Add id2, seq2, and target_file columns of CS_Dz.csv to model-prefixed feature_set_predicted.csv
        cs_dz_file_path = os.path.join(args.output_dir, cs_dz_file)
        df_cs_dz = pd.read_csv(cs_dz_file_path)
        feature_set_predicted_file = f"{args.default_train_file}_feature_set_predicted.csv"
        feature_set_predicted_path = os.path.join(args.output_dir, feature_set_predicted_file)
        df_feature_set = pd.read_csv(feature_set_predicted_path)
        # Insert target_file after seq2 if present, else after id2
        if 'seq2' in df_cs_dz.columns:
            cols = ['id2', 'seq2', 'target_file'] + [col for col in df_feature_set.columns if col not in ['id2', 'seq2', 'target_file']]
            df_feature_set = pd.concat([df_cs_dz, df_feature_set], axis=1)
            df_feature_set = df_feature_set.loc[:,~df_feature_set.columns.duplicated()]
            df_feature_set = df_feature_set[cols]
        else:
            cols = ['id2', 'target_file'] + [col for col in df_feature_set.columns if col not in ['id2', 'target_file']]
            df_feature_set = pd.concat([df_cs_dz, df_feature_set], axis=1)
            df_feature_set = df_feature_set.loc[:,~df_feature_set.columns.duplicated()]
            df_feature_set = df_feature_set[cols]
        df_feature_set.to_csv(feature_set_predicted_path, index=False)
        report_file_status(feature_set_predicted_path, "Updated feature set predicted")

        # Sort feature_set_predicted.csv by y_pred (1 then 0) and reliability_score (high to low)
        df_feature_set = pd.read_csv(feature_set_predicted_path)
        df_feature_set = df_feature_set.sort_values(by=['y_pred', 'reliability_score'], ascending=[False, False])
        df_feature_set.to_csv(feature_set_predicted_path, index=False)
        report_file_status(feature_set_predicted_path, "Sorted feature set predicted")
    elif args.user_train_file:
        model_name = args.user_train_file

        # File paths
        target_file = f"{model_name}_target.csv"

        # Validate that the required target file exists
        if not os.path.exists(target_file):
            print(f"Error: Required file '{target_file}' does not exist.")
            sys.exit(1)

        # Calculate mean and std for all_generated_merged_num.csv
        df_generated = pd.read_csv("all_generated_merged_num.csv")
        mean_std = df_generated.describe().loc[['mean', 'std']]
        mean_std_file = f"{model_name}_user_train_statistics.csv"
        mean_std.to_csv(mean_std_file)
        report_file_status(mean_std_file, "User train statistics")

        # Standardize numerical columns in all_generated_merged_num.csv
        df_standardized = df_generated.copy()
        for column in df_standardized.columns:
            if column != 'id2':
                df_standardized[column] = (df_standardized[column] - mean_std.loc['mean', column]) / mean_std.loc['std', column]

        # Retain id2 column and save standardized data
        df_standardized = df_standardized[['id2'] + [col for col in df_standardized.columns if col != 'id2']]
        standardized_file = f"{model_name}_standardized_user_train.csv"
        df_standardized.to_csv(standardized_file, index=False)
        report_file_status(standardized_file, "Standardized user train")

        # Balance target file
        df_target = pd.read_csv(target_file)
        np.random.seed(89273554)
        df_balanced = df_target.groupby('Y', group_keys=False).apply(lambda x: x.sample(df_target['Y'].value_counts().min()))
        balanced_file = f"{model_name}_balanced_target.csv"
        df_balanced.to_csv(balanced_file, index=False)
        report_file_status(balanced_file, "Balanced target")

        df_non_balanced = df_target[~df_target.index.isin(df_balanced.index)]
        non_balanced_file = f"{model_name}_non_balanced_target.csv"
        df_non_balanced.to_csv(non_balanced_file, index=False)
        report_file_status(non_balanced_file, "Non-balanced target")

        # Merge standardized_user_train with balanced_target
        df_merged_train = pd.merge(df_standardized, df_balanced, on='id2')
        merged_train_file = f"{model_name}_user_ML_train.csv"
        df_merged_train.to_csv(merged_train_file, index=False)
        report_file_status(merged_train_file, "User ML train")

        # Merge standardized_user_train with non_balanced_target
        df_merged_test = pd.merge(df_standardized, df_non_balanced, on='id2')
        merged_test_file = f"{model_name}_user_ML_test.csv"
        df_merged_test.to_csv(merged_test_file, index=False)
        report_file_status(merged_test_file, "User ML test")

        # Define the single feature set
        feature_set = ['Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 'E_3', 'seedNumber_3', 'pumin1_4d', 'pumin5_8d']

        # Ensure df_train is initialized before use
        df_train = pd.read_csv(merged_train_file)

        # Extract feature sets from user_ML_train and include 'Y' column
        feature_set_with_y = feature_set + ['Y']

        feature_set_file = f"{model_name}_user_ML_train_feature_set.csv"
        df_train[feature_set_with_y].to_csv(feature_set_file, index=False)
        report_file_status(feature_set_file, "User ML train feature set")

        # Standardize user_ML_test.csv
        df_test = pd.read_csv(merged_test_file)
        df_test_standardized = (df_test - mean_std.loc['mean']) / mean_std.loc['std']
        standardized_test_file = f"{model_name}_standardized_user_ML_test.csv"
        df_test_standardized.to_csv(standardized_test_file, index=False)
        report_file_status(standardized_test_file, "Standardized user ML test")

        # Extract feature set from user_ML_test.csv
        feature_set_file = f"{model_name}_user_ML_test_feature_set.csv"
        df_test = pd.read_csv(merged_test_file)
        df_test[feature_set].to_csv(feature_set_file, index=False)
        report_file_status(feature_set_file, "User ML test feature set")

        # Train and save SVM model for feature set
        train_and_save_svm(f"{model_name}_user_ML_train_feature_set.csv", model_name, "user_train_feature_set")

        # Add target_file column to prediction output if generated
        user_pred_file = os.path.join(args.output_dir, f"{model_name}_feature_set_predicted.csv")
        if os.path.exists(user_pred_file):
            pred_df = pd.read_csv(user_pred_file)
            all_gen = pd.read_csv(os.path.join(args.output_dir, "all_generated_merged_num.csv"))
            id2_to_target = dict(zip(all_gen['id2'], all_gen['target_file'])) if 'target_file' in all_gen.columns else {}
            if 'id2' in pred_df.columns:
                pred_df.insert(1, 'target_file', pred_df['id2'].map(id2_to_target) if id2_to_target else None)
                pred_df.to_csv(user_pred_file, index=False)
                print(f"Added target_file column to {user_pred_file}")
    else:
        print("Error: Either --default_train_file or --user_train_file must be provided.")
        sys.exit(1)

def predict_with_confidence(model, X, y_true=None):
    """
    Returns predictions, reliability score (predict_proba),
    decision_function (distance to boundary),
    entropy of probabilities, and Brier score (if y_true provided).
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    reliability_score = y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba[:, 0]
    decision_score = model.decision_function(X)
    entropies = entropy(y_proba.T, base=2)  # entropy per sample
    brier = None
    if y_true is not None:
        brier = brier_score_loss(y_true, reliability_score)
    return y_pred, reliability_score, decision_score, entropies, brier

def read_fasta_sequence(fasta_file):
    """Read the first sequence from a fasta file as a single string."""
    seq = ''
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    seq += line.strip()
    except Exception as e:
        print(f"Error reading fasta file {fasta_file}: {e}")
    return seq

def run_rnafold_and_get_structure(sequence):
    """
    Run RNAfold on the given sequence and return (sequence, dot_bracket, mfe).
    """
    import subprocess
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as fasta_file:
        fasta_file.write(">seq\n" + sequence + "\n")
        fasta_file.flush()
        try:
            result = subprocess.run(["RNAfold", fasta_file.name], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 3:
                seq = lines[1].strip()
                struct_line = lines[2].strip()
                if ' ' in struct_line:
                    dot_bracket, mfe = struct_line.split(' ', 1)
                    mfe = mfe.strip('() ').replace('kcal/mol','')
                else:
                    dot_bracket, mfe = struct_line, ''
                return seq, dot_bracket, mfe
        except Exception as e:
            print(f"Error running RNAfold: {e}")
    return sequence, '', ''

def dotbracket_to_pairs(dot_bracket):
    """
    Convert dot-bracket notation to list of base pairs (1-based).
    """
    stack = []
    pairs = []
    for i, c in enumerate(dot_bracket):
        if c == '(': 
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                pairs.append((j+1, i+1))
    return pairs

def process_top10_rna_structures(predicted_file, targets, output_dir):
    """
    For the top 10 by reliability_score in predicted_file, run RNAfold and RNAplot for each unique target_file.
    Save PNGs named after the target_file in output_dir.
    """
    import pandas as pd
    import subprocess
    import os

    # Read predictions and get top 10 by reliability_score
    df = pd.read_csv(predicted_file)
    if 'reliability_score' not in df.columns or 'target_file' not in df.columns:
        print(f"Error: Columns 'reliability_score' or 'target_file' not found in {predicted_file}.")
        return
    top10 = df.sort_values(by='reliability_score', ascending=False).head(10)
    unique_targets = top10['target_file'].unique()

    # Map target_file to fasta path
    target_map = {os.path.splitext(os.path.basename(t))[0]: t for t in targets}

    for target_name in unique_targets:
        # Remove extension for matching
        base_name = os.path.splitext(os.path.basename(str(target_name)))[0]
        fasta_path = target_map.get(base_name)
        if not fasta_path or not os.path.exists(fasta_path):
            print(f"Warning: FASTA file for target '{target_name}' not found.")
            continue
        # Read sequence
        seq = read_fasta_sequence(fasta_path)
        if not seq:
            print(f"Warning: No sequence found in {fasta_path}.")
            continue
        # Run RNAfold
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fa') as fasta_file:
            fasta_file.write(f">{base_name}\n{seq}\n")
            fasta_file.flush()
            try:
                result = subprocess.run(["RNAfold", fasta_file.name], capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 3:
                    struct_line = lines[2].strip()
                    if ' ' in struct_line:
                        dot_bracket, mfe = struct_line.split(' ', 1)
                        mfe = mfe.strip('() ').replace('kcal/mol','')
                    else:
                        dot_bracket, mfe = struct_line, ''
                else:
                    print(f"Warning: Unexpected RNAfold output for {base_name}.")
                    continue
            except Exception as e:
                print(f"Error running RNAfold for {base_name}: {e}")
                continue
        # Write structure for RNAplot
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.str') as struct_file:
            struct_file.write(f">{base_name}\n{seq}\n{dot_bracket}\n")
            struct_file.flush()
            ps_path = os.path.join(output_dir, f"{base_name}_ss.ps")
            png_path = os.path.join(output_dir, f"{base_name}.png")
            try:
                # Run RNAplot (no -o or -t options)
                subprocess.run(["RNAplot", struct_file.name], cwd=output_dir, check=True)
                # RNAplot outputs <basename>_ss.ps in the working dir
                if os.path.exists(ps_path):
                    # Convert .ps to .png using ImageMagick's convert
                    try:
                        subprocess.run(["convert", ps_path, png_path], check=True)
                        print(f"Saved RNA structure plot: {png_path}")
                        os.remove(ps_path)
                    except Exception as e:
                        print(f"Error converting {ps_path} to PNG: {e}")
                        print(f"PS file left at: {ps_path}")
                else:
                    print(f"RNAplot did not produce expected PS for {base_name}.")
            except Exception as e:
                print(f"Error running RNAplot for {base_name}: {e}")

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--targets', required=True, nargs='+', help="Path to one or more FASTA files")
        parser.add_argument('--params', required=True, help="Path to the CSV file containing LA, RA, CS, temperature, and core")
        parser.add_argument('--feature_mode', required=True, choices=['default', 'target_screen', 'target_check', 'specific_query'], help="Mode of operation")
        parser.add_argument('--specific_csv', help="CSV file for specific_query mode")
        parser.add_argument('--default_train_file', help="Prefix of the model name for default training")
        parser.add_argument('--user_train_file', help="Path to user-provided training file")
        parser.add_argument('--output_dir', help="Directory to save all outputs", default=os.getcwd())
        args = parser.parse_args()

        # Debugging: Print parsed arguments
        print("Parsed arguments:", args)

        # Validate each target file in the list
        for target in args.targets:
            if not os.path.exists(target):
                print(f"Error: Target file '{target}' does not exist.")
                sys.exit(1)

        if not os.path.exists(args.params):
            print(f"Error: Parameters file '{args.params}' does not exist.")
            sys.exit(1)

        if args.feature_mode == 'specific_query' and args.specific_csv and not os.path.exists(args.specific_csv):
            print(f"Error: Specific CSV file '{args.specific_csv}' does not exist.")
            sys.exit(1)

        # Debugging: Print mode-specific logic
        print(f"Running in mode: {args.feature_mode}")
        if args.specific_csv:
            print(f"Using specific CSV: {args.specific_csv}")

        # Create parameters.cfg if it doesn't exist
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.cfg")):
            create_cfg_file(args.params)

        # Execute Feature processing, removing the placeholder print
        train(args)

        # After all predictions and outputs are generated, process top 10 reliability_score
        # Determine the predicted file name
        if args.default_train_file:
            predicted_file = os.path.join(args.output_dir, f"{args.default_train_file}_feature_set_predicted.csv")
        elif args.user_train_file:
            predicted_file = os.path.join(args.output_dir, f"{args.user_train_file}_feature_set_predicted.csv")
        else:
            predicted_file = os.path.join(args.output_dir, "feature_set_predicted.csv")
        if os.path.exists(predicted_file):
            process_top10_rna_structures(predicted_file, args.targets, args.output_dir)
        else:
            print(f"Predicted file {predicted_file} not found. Skipping RNA structure plotting.")
    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()