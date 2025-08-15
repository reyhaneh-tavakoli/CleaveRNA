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
import pickle
import tempfile
import time
from tqdm import tqdm
import threading

# Progress tracking and UI enhancements
class ProgressTracker:
    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.pbar = None
        
    def start(self, description="Processing"):
        """Initialize progress bar"""
        self.pbar = tqdm(total=self.total_steps, desc=description, 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
    def update(self, steps=1, description=None):
        """Update progress bar"""
        if self.pbar:
            self.current_step += steps
            if description:
                self.pbar.set_description(description)
            self.pbar.update(steps)
            
    def finish(self):
        """Close progress bar"""
        if self.pbar:
            self.pbar.close()
            
    def step_success(self, step_name, duration=None):
        """Report successful completion of a step with sticker"""
        success_stickers = ['✅', '🎉', '✨', '🚀', '⭐', '💫']
        sticker = np.random.choice(success_stickers)
        if duration:
            print(f"{sticker} {step_name} completed successfully in {duration:.2f}s")
        else:
            print(f"{sticker} {step_name} completed successfully")
            
    def estimate_remaining_time(self):
        """Estimate remaining time based on completed steps"""
        if self.current_step > 0:
            elapsed = time.time() - self.start_time
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            return estimated_remaining
        return 0

def predict_execution_time(args):
    """Predict estimated execution time based on input parameters"""
    base_time = 10  # Base processing time in seconds
    
    # Factor in the number of target files
    if args.targets:
        file_factor = len(args.targets) * 5  # 5 seconds per target file
    elif args.train_mode:
        file_factor = len(args.train_mode) * 8  # 8 seconds per training file
    else:
        file_factor = 5
    
    # Factor in feature mode complexity
    feature_complexity = {
        'default': 1.0,
        'target_screen': 1.5,
        'target_check': 1.2,
        'specific_query': 0.8
    }
    
    mode_factor = feature_complexity.get(args.feature_mode, 1.0)
    
    # Factor in prediction vs training mode
    if args.prediction_mode:
        mode_time = 20  # Prediction mode takes longer
    elif args.train_mode:
        mode_time = 15  # Training mode
    else:
        mode_time = 10
    
    estimated_time = (base_time + file_factor + mode_time) * mode_factor
    return estimated_time

def format_time(seconds):
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def create_cfg_file(params_file, output_dir='.'):
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
    # Save the parameters.cfg file in the output directory
    os.makedirs(output_dir, exist_ok=True)
    cfg_file_path = os.path.join(output_dir, "parameters.cfg")
    with open(cfg_file_path, "w") as cfg_file:
        cfg_file.write(cfg_content)
    # Suppress the creation message
    return cfg_file_path

def report_file_status(file_path, description):
    error_stickers = ['❌', '⚠️', '💥']
    
    if os.path.exists(file_path):
        # Only show errors, suppress success messages
        pass
    else:
        sticker = np.random.choice(error_stickers)
        print(f"{sticker} Error: {description} was not generated.")

def report_empty_file(file_path, description):
    warning_stickers = ['⚠️', '🚨', '⏰']
    
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == 0:
            sticker = np.random.choice(warning_stickers)
            print(f"{sticker} Warning: {description} is empty and will be skipped: {file_path}.")
            return True
    else:
        sticker = np.random.choice(warning_stickers)
        print(f"{sticker} Warning: {description} does not exist: {file_path}.")
        return True
    return False

def train_and_save_svm(train_data_path, model_name, feature_set_name, progress_tracker=None):
    step_start_time = time.time()
    # Suppressed SVM training message

    if progress_tracker:
        progress_tracker.update(2, f"Loading training data for {model_name}")

    df = pd.read_csv(train_data_path)
    if 'Y' not in df.columns:
        raise ValueError("Training data must include a target column named 'Y'.")

    # Check class balance and balance if needed
    y_counts = df['Y'].value_counts()
    if len(y_counts) == 2 and y_counts.iloc[0] != y_counts.iloc[1]:
        # Suppressed balance detection message
        if progress_tracker:
            progress_tracker.update(1, "Balancing classes")
        min_count = y_counts.min()
        df_balanced = df.groupby('Y', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
        df = df_balanced
        # Suppressed balance success message
    elif len(y_counts) == 2:
        # Suppressed already balanced message
        pass
    else:
        print("⚠️ Warning: Only one class present in target column. SVM training may fail.")

    if progress_tracker:
        progress_tracker.update(2, "Preprocessing features")

    X_df = df.drop(columns=['Y'])
    y = df['Y']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X_df)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if progress_tracker:
        progress_tracker.update(3, f"Training SVM model for {model_name}")

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
    
    step_duration = time.time() - step_start_time
    if progress_tracker:
        progress_tracker.step_success(f"SVM model training for {model_name}", step_duration)
        progress_tracker.update(2, "Running cross-validation")
    else:
        # Suppressed model save message
        pass

    # Cross-validation with suppressed output
    perform_cross_validation(X, y, model_name, feature_set_name, progress_tracker)

def perform_cross_validation(X, y, model_name, feature_set_name, progress_tracker=None):
    step_start_time = time.time()
    # Suppressed cross-validation output
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(C=10, gamma='auto', kernel='rbf', probability=True, random_state=42)

    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    proba_results = []  # Store predict_proba results for each fold

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if progress_tracker:
            progress_tracker.update(1, f"Cross-validation fold {fold+1}/5")
        
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

    # Suppress cross-validation results output

    # Save cross-validation results to a CSV file
    metrics_file = f"{model_name}_ML_metrics_{feature_set_name}.csv"
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Mean': [np.mean(scores['accuracy']), np.mean(scores['precision']), np.mean(scores['recall']), np.mean(scores['f1'])],
        'Std': [np.std(scores['accuracy']), np.std(scores['precision']), np.std(scores['recall']), np.std(scores['f1'])]
    })
    metrics_df.to_csv(metrics_file, index=False)
    
    step_duration = time.time() - step_start_time
    if progress_tracker:
        progress_tracker.step_success(f"Cross-validation for {feature_set_name}", step_duration)
    
    report_file_status(metrics_file, f"ML metrics for {feature_set_name}")

def train(args):
    # Initialize progress tracking
    estimated_time = predict_execution_time(args)
    print(f"\n🚀 Starting CleaveRNA analysis...")
    print(f"⏱️ Estimated execution time: {format_time(estimated_time)}")
    
    # Determine total steps based on mode
    if args.prediction_mode:
        total_steps = 25  # More steps for prediction mode
    elif args.train_mode:
        total_steps = 15  # Fewer steps for train mode
    else:
        total_steps = 10
    
    progress = ProgressTracker(total_steps)
    progress.start("CleaveRNA Processing")
    
    try:
        # Step 1: Setup output directory
        step_start = time.time()
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"📁 Output directory set to: {args.output_dir}")
        progress.update(1, "Setting up output directory")
        progress.step_success("Output directory setup", time.time() - step_start)

        # Skip initial Feature.py run for train_mode as it handles its own feature generation
        if not args.train_mode:
            # Step 2: Run Feature.py
            step_start = time.time()
            print("🔧 Running Feature.py to generate 'all_generated_merged_num.csv'...")
            targets_arg = ','.join(args.targets) if args.targets else ''
            feature_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Feature.py'))
            feature_command = f"python3 {feature_script_path} --targets {targets_arg} --params {args.params} --feature_mode {args.feature_mode} --output_dir {args.output_dir}"
            
            progress.update(2, "Executing Feature.py")
            try:
                subprocess.run(feature_command, shell=True, check=True, cwd=args.output_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                progress.step_success("Feature.py execution", time.time() - step_start)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error: Feature.py execution failed with error: {e}")
                sys.exit(1)

            # Step 3: Verify output file generation
            step_start = time.time()
            progress.update(1, "Verifying output file generation")
            retries = 5
            output_file_path = os.path.join(args.output_dir, "all_generated_merged_num.csv")
            for attempt in range(retries):
                if os.path.exists(output_file_path):
                    progress.step_success("Output file verification", time.time() - step_start)
                    break
                else:
                    print(f"🔄 Attempt {attempt + 1}/{retries}: 'all_generated_merged_num.csv' not found. Retrying...")
                    time.sleep(2)
            else:
                print(f"❌ Error: 'all_generated_merged_num.csv' was not generated after multiple attempts.")
                sys.exit(1)

        if args.prediction_mode:
            model_name = args.model_name
            target_file = args.ML_target
            default_merged_file = args.prediction_mode

            # Step 4: Validate required files
            step_start = time.time()
            progress.update(1, "Validating required files")
            if not os.path.exists(default_merged_file) or not os.path.exists(target_file):
                print(f"❌ Error: Required files '{default_merged_file}' or '{target_file}' do not exist.")
                sys.exit(1)
            progress.step_success("File validation", time.time() - step_start)

            # Step 5: Check result files
            step_start = time.time()
            progress.update(1, "Checking result files")
            result_files = [
                f"{model_name}_Results_with_region.csv",
                f"{model_name}_Results_without_region.csv",
                f"{model_name}_Results_pairwise.csv"
            ]
            for result_file in result_files:
                if report_empty_file(result_file, "Result file"):
                    continue
            progress.step_success("Result file check", time.time() - step_start)

            # Step 6: Calculate statistics
            step_start = time.time()
            progress.update(2, "Calculating training statistics")
            df_default = pd.read_csv(default_merged_file)
            # Only calculate statistics for numeric columns
            numeric_columns = df_default.select_dtypes(include=[np.number]).columns
            mean_std = df_default[numeric_columns].describe().loc[['mean', 'std']]
            mean_std_file = f"{model_name}_default_train_statistics.csv"
            mean_std.to_csv(mean_std_file)
            report_file_status(mean_std_file, "Default train statistics")
            progress.step_success("Statistics calculation", time.time() - step_start)

            # Step 7: Standardize data
            step_start = time.time()
            progress.update(2, "Standardizing training data")
            df_standardized = df_default.copy()
            # Only standardize numeric columns, skip text columns like 'id2', 'seq2', 'target_file'
            non_numeric_columns = ['id2', 'seq2', 'target_file']
            for column in df_standardized.columns:
                if column not in non_numeric_columns and column in mean_std.columns:
                    try:
                        df_standardized[column] = (df_standardized[column] - mean_std.loc['mean', column]) / mean_std.loc['std', column]
                    except (KeyError, TypeError):
                        # Skip columns that can't be standardized
                        continue
            standardized_file = f"{model_name}_standardized_default_train.csv"
            df_standardized.to_csv(standardized_file, index=False)
            report_file_status(standardized_file, "Standardized default train")
            progress.step_success("Data standardization", time.time() - step_start)

            # Step 8: Balance target data
            step_start = time.time()
            progress.update(2, "Balancing target data")
            df_target = pd.read_csv(target_file)
            np.random.seed(89273554)
            df_balanced = df_target.groupby('Y', group_keys=False).apply(
                lambda x: x.sample(df_target['Y'].value_counts().min(), random_state=89273554)
            )
            balanced_file = f"{model_name}_balanced_target.csv"
            df_balanced.to_csv(balanced_file, index=False)
            report_file_status(balanced_file, "Balanced target")

            # Skip generating non-balanced target and default ML test files
            progress.step_success("Target data balancing", time.time() - step_start)

            # Step 9: Merge training data
            step_start = time.time()
            progress.update(2, "Merging training datasets")
            df_merged_train = pd.merge(df_standardized, df_balanced, on='id2')
            merged_train_file = f"{model_name}_default_ML_train.csv"
            df_merged_train.to_csv(merged_train_file, index=False)
            report_file_status(merged_train_file, "Default ML train")

            # Skip generating default ML test file
            progress.step_success("Dataset merging", time.time() - step_start)

            # Step 10: Prepare feature sets
            step_start = time.time()
            progress.update(1, "Preparing feature sets")
            feature_set = ['Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 'E_3', 'seedNumber_3', 'pumin1_4d', 'pumin5_8d']
            df_train = pd.read_csv(merged_train_file)
            feature_set_with_y = feature_set + ['Y']
            feature_set_file = f"{model_name}_default_ML_train_feature_set.csv"
            df_train[feature_set_with_y].to_csv(feature_set_file, index=False)
            report_file_status(feature_set_file, "Default ML train feature set")
            progress.step_success("Feature set preparation", time.time() - step_start)

            # Step 11: Standardize generated data
            step_start = time.time()
            progress.update(2, "Standardizing generated data")
            mean_std_file = f"{model_name}_default_train_statistics.csv"
            mean_std = pd.read_csv(mean_std_file, index_col=0)
            df_generated = pd.read_csv("all_generated_merged_num.csv")
            df_standardized_generated = df_generated.copy()

            # Only standardize columns that exist in both dataframes and are numeric
            for column in mean_std.columns:
                if column in df_standardized_generated.columns:
                    try:
                        df_standardized_generated[column] = (df_standardized_generated[column] - mean_std.loc['mean', column]) / mean_std.loc['std', column]
                    except (KeyError, TypeError):
                        # Skip columns that can't be standardized
                        continue

            # Keep essential columns plus standardized numeric columns
            essential_columns = ['id2', 'seq2', 'target_file']
            columns_to_keep = [col for col in essential_columns if col in df_standardized_generated.columns] + list(mean_std.columns)
            # Remove duplicates while preserving order
            columns_to_keep = list(dict.fromkeys(columns_to_keep))
            # Only keep columns that actually exist in the dataframe
            columns_to_keep = [col for col in columns_to_keep if col in df_standardized_generated.columns]
            df_standardized_generated = df_standardized_generated[columns_to_keep]
            
            standardized_generated_file = "standardized_all_generated_merged_num.csv"
            df_standardized_generated.to_csv(standardized_generated_file, index=False)
            report_file_status(standardized_generated_file, "Standardized generated merged num")

            generated_feature_set_file = "generated_ML_test_feature_set.csv"
            # Only select feature columns that exist in the dataframe
            available_features = [col for col in feature_set if col in df_standardized_generated.columns]
            if available_features:
                df_standardized_generated[available_features].to_csv(generated_feature_set_file, index=False)
            else:
                print("⚠️ Warning: No feature set columns found in standardized data")
            report_file_status(generated_feature_set_file, "Generated ML test feature set")
            progress.step_success("Generated data standardization", time.time() - step_start)

            # Step 12-17: Train SVM model (progress handled within function)
            train_and_save_svm(f"{model_name}_default_ML_train_feature_set.csv", model_name, "default_train_feature_set", progress)

            # Step 18: Make predictions
            step_start = time.time()
            progress.update(2, "Making predictions")
            # Suppressed prediction processing message
            pickle_file = f"{model_name}-default_train_feature_set-SVM.pkl"
            test_file = "generated_ML_test_feature_set.csv"
            output_file = f"{model_name}_feature_set_predicted.csv"
            model_file = os.path.join(args.output_dir, pickle_file)
            output_path = os.path.join(args.output_dir, output_file)
            
            if not os.path.exists(model_file):
                print(f"⚠️ {model_file} not found. Skipping.")
            else:
                with open(model_file, 'rb') as f:
                    model_bundle = pickle.load(f)
                model = model_bundle['model']
                imputer = model_bundle['imputer']
                feature_columns = model_bundle['feature_columns']
                test_file_path = os.path.join(args.output_dir, test_file)
                
                if not os.path.exists(test_file_path):
                    print(f"⚠️ {test_file_path} not found. Skipping.")
                else:
                    df_test = pd.read_csv(test_file_path)
                    available_columns = [col for col in feature_columns if col in df_test.columns]
                    if not available_columns:
                        print(f"⚠️ No matching feature columns found in {test_file_path}. Skipping.")
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
                        
                        Classification_score, reliability_score, decision_score, _ = predict_with_confidence(model, X_std, y_true)
                        predict_proba = model.predict_proba(X_std)[:, 1] if model.predict_proba(X_std).shape[1] >= 2 else model.predict_proba(X_std)[:, 0]
                        margin = np.abs(decision_score)
                        combined_score = reliability_score * margin
                        
                        result_df = pd.DataFrame({
                            'id2': df_test['id2'] if 'id2' in df_test.columns else range(len(df_test)),
                            'Classification_score': Classification_score,
                            'reliability_score': reliability_score,
                            'predict_proba': predict_proba,
                            'decision_score': decision_score,
                            'margin': margin,
                            'combined_score': combined_score
                        })
                        
                        all_gen = pd.read_csv(os.path.join(args.output_dir, "all_generated_merged_num.csv"))
                        id2_to_target = dict(zip(all_gen['id2'], all_gen['target_file'])) if 'target_file' in all_gen.columns else {}
                        result_df.insert(1, 'target_file', result_df['id2'].map(id2_to_target) if id2_to_target else None)
                        result_df.to_csv(output_path, index=False)
                        # Suppress success message
            
            progress.step_success("Predictions", time.time() - step_start)

            # Step 19: Process CS_Dz mapping - Extract id2, seq2, and target_file from CS_Dz file
            step_start = time.time()
            progress.update(1, "Processing CS_Dz mapping")
            cs_dz_path = os.path.join(args.output_dir, "CS_Dz.csv")
            if os.path.exists(cs_dz_path) and os.path.exists(output_path):
                df_cs_dz = pd.read_csv(cs_dz_path)
                result_df = pd.read_csv(output_path)
                if len(result_df) == len(df_cs_dz):
                    # Extract id2, seq2, and target_file from CS_Dz file and override in predicted results
                    result_df['id2'] = df_cs_dz['id2'].astype(str).values
                    if 'seq2' in df_cs_dz.columns:
                        result_df['seq2'] = df_cs_dz['seq2'].values
                    if 'target_file' in df_cs_dz.columns:
                        result_df['target_file'] = df_cs_dz['target_file'].values
                    
                    # Reorder columns to put id2, seq2, target_file at the beginning
                    cols = list(result_df.columns)
                    priority_cols = ['id2', 'seq2', 'target_file']
                    ordered_cols = []
                    
                    # Add priority columns first (if they exist)
                    for col in priority_cols:
                        if col in cols:
                            ordered_cols.append(col)
                            cols.remove(col)
                    
                    # Add remaining columns
                    ordered_cols.extend(cols)
                    result_df = result_df[ordered_cols]
                    result_df.to_csv(output_path, index=False)
                    print(f"✅ Updated predicted file with id2, seq2, and target_file from CS_Dz.csv")
                else:
                    print(f"⚠️ Warning: Row count mismatch between predicted results ({len(result_df)}) and CS_Dz file ({len(df_cs_dz)})")
            else:
                if not os.path.exists(cs_dz_path):
                    print(f"⚠️ Warning: CS_Dz.csv not found at {cs_dz_path}")
                if not os.path.exists(output_path):
                    print(f"⚠️ Warning: Predicted output file not found at {output_path}")
            progress.step_success("CS_Dz mapping", time.time() - step_start)

            # Step 20: Save CS_Dz file
            step_start = time.time()
            progress.update(1, "Saving CS_Dz file")
            df_all_generated = pd.read_csv("all_generated_merged_num.csv")
            cs_dz_file = "CS_Dz.csv"
            if 'target_file' in df_all_generated.columns:
                df_all_generated[['id2', 'seq2', 'target_file']].to_csv(cs_dz_file, index=False)
            else:
                fasta_targets = args.targets if isinstance(args.targets, list) else args.targets.split(',')
                df_all_generated['target_file'] = None
                if len(fasta_targets) == len(df_all_generated):
                    df_all_generated['target_file'] = fasta_targets
                else:
                    df_all_generated['target_file'] = fasta_targets[0] if fasta_targets else None
                df_all_generated[['id2', 'seq2', 'target_file']].to_csv(cs_dz_file, index=False)
            report_file_status(cs_dz_file, "CS_Dz file")
            progress.step_success("CS_Dz file creation", time.time() - step_start)

            # Step 21: Final processing
            step_start = time.time()
            progress.update(2, "Final result processing")
            cs_dz_file_path = os.path.join(args.output_dir, cs_dz_file)
            feature_set_predicted_path = os.path.join(args.output_dir, f"{model_name}_feature_set_predicted.csv")
            if os.path.exists(cs_dz_file_path) and os.path.exists(feature_set_predicted_path):
                df_cs_dz = pd.read_csv(cs_dz_file_path)
                df_feature_set = pd.read_csv(feature_set_predicted_path)
                keep_cols = [col for col in ['id2', 'seq2', 'target_file', 'Classification_score', 'reliability_score', 'decision_score', 'brier_score'] if col in df_feature_set.columns]
                df_feature_set = df_feature_set[keep_cols]
                cols_to_remove = [col for col in df_feature_set.columns if col in ['id2.1', 'seq2.1', 'target_file.1'] or col.startswith('id2.') or col.startswith('seq2.') or col.startswith('target_file.')]
                if cols_to_remove:
                    df_feature_set = df_feature_set.drop(columns=cols_to_remove)
                df_feature_set.to_csv(feature_set_predicted_path, index=False)
                report_file_status(feature_set_predicted_path, "Updated feature set predicted")
                
                df_feature_set = pd.read_csv(feature_set_predicted_path)
                df_feature_set = df_feature_set.sort_values(by=['Classification_score', 'reliability_score'], ascending=[False, False])
                df_feature_set.to_csv(feature_set_predicted_path, index=False)
                report_file_status(feature_set_predicted_path, "Sorted feature set predicted")
            progress.step_success("Final result processing", time.time() - step_start)

        elif args.train_mode:
            model_name = args.model_name
            user_train_files = args.train_mode
            
            # Step 4: Generate user features
            step_start = time.time()
            progress.update(3, "Generating user training features")
            feature_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Feature.py'))
            user_merged_file = os.path.join(args.output_dir, f"{model_name}_user_merged_num.csv")
            user_train_command = f"python3 {feature_script_path} --targets {','.join(user_train_files)} --params {args.params} --feature_mode default --output_dir {args.output_dir}"
            try:
                subprocess.run(user_train_command, shell=True, check=True, cwd=args.output_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                progress.step_success("User feature generation", time.time() - step_start)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error: Feature.py execution failed for user_train_file with error: {e}")
                sys.exit(1)
            
            # Step 5: Rename output file
            step_start = time.time()
            progress.update(2, "Finalizing user training data")
            default_generated_file = os.path.join(args.output_dir, "all_generated_merged_num.csv")
            if os.path.exists(default_generated_file):
                os.rename(default_generated_file, user_merged_file)
                # Suppressed rename message
                progress.step_success("User train mode completion", time.time() - step_start)
                # Suppressed completion message
            else:
                print(f"❌ Error: {default_generated_file} not found after Feature.py run.")
                sys.exit(1)
        else:
            print("❌ Error: Either --prediction_mode or --train_mode must be provided.")
            sys.exit(1)

        # Final completion
        progress.update(progress.total_steps - progress.current_step, "Finalizing")
        total_time = time.time() - progress.start_time
        progress.finish()
        
        print(f"\n🎊 CleaveRNA analysis completed successfully!")
        print(f"⏱️ Total execution time: {format_time(total_time)}")
        print(f"📊 All output files have been generated in: {args.output_dir}")
        
    except Exception as e:
        progress.finish()
        print(f"\n💥 An error occurred during processing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def predict_with_confidence(model, X, y_true=None):
    """
    Returns predictions, reliability score (predict_proba),
    decision_function (distance to boundary),
    and Brier score (if y_true provided).
    """
    Classification_score = model.predict(X)
    y_proba = model.predict_proba(X)
    reliability_score = y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba[:, 0]
    decision_score = model.decision_function(X)
    brier = None
    if y_true is not None:
        brier = brier_score_loss(y_true, reliability_score)
    return Classification_score, reliability_score, decision_score, brier

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

def main():
    try:
        print("\n" + "="*60)
        print("🧬 CleaveRNA Analysis Tool 🧬")
        print("="*60)
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--targets', nargs='+', help="Path to one or more FASTA files (required for --prediction_mode)")
        parser.add_argument('--params', help="Path to the CSV file containing LA, RA, CS, temperature, and core")
        parser.add_argument('--feature_mode', choices=['default', 'target_screen', 'target_check', 'specific_query'], help="Mode of operation")
        parser.add_argument('--prediction_mode', help="CSV file for prediction mode")
        parser.add_argument('--train_mode', nargs='+', help="FASTA file(s) for user train mode")
        parser.add_argument('--model_name', required=True, help="Model name for output file naming")
        parser.add_argument('--output_dir', help="Directory to save all outputs", default=os.getcwd())
        parser.add_argument('--specific_csv', help="CSV file for specific_query mode", default=None)
        parser.add_argument('--ML_target', help="CSV file containing target labels for ML training (required for --prediction_mode)")
        args = parser.parse_args()

        # Show initial configuration
        print(f"🔧 Configuration loaded:")
        print(f"   Model name: {args.model_name}")
        print(f"   Feature mode: {args.feature_mode}")
        if args.prediction_mode:
            print(f"   Mode: Prediction")
            print(f"   Targets: {len(args.targets) if args.targets else 0} files")
        elif args.train_mode:
            print(f"   Mode: Training")
            print(f"   Training files: {len(args.train_mode)} files")
        print(f"   Output directory: {args.output_dir}")

        # Validate that one of the train modes is specified
        if not args.prediction_mode and not args.train_mode:
            print("❌ Error: Either --prediction_mode or --train_mode must be provided.")
            sys.exit(1)

        # Validate ML_target is provided for prediction_mode
        if args.prediction_mode and not args.ML_target:
            print("❌ Error: --ML_target is required when using --prediction_mode.")
            sys.exit(1)

        # Validate targets is provided for prediction_mode
        if args.prediction_mode and not args.targets:
            print("❌ Error: --targets is required when using --prediction_mode.")
            sys.exit(1)

        # Validate each target file in the list (only for prediction_mode)
        if args.targets:
            for target in args.targets:
                if not os.path.exists(target):
                    print(f"❌ Error: Target file '{target}' does not exist.")
                    sys.exit(1)

        if not os.path.exists(args.params):
            print(f"❌ Error: Parameters file '{args.params}' does not exist.")
            sys.exit(1)

        if args.feature_mode == 'specific_query' and args.specific_csv and not os.path.exists(args.specific_csv):
            print(f"❌ Error: Specific CSV file '{args.specific_csv}' does not exist.")
            sys.exit(1)

        # Create parameters.cfg if it doesn't exist
        cfg_path = os.path.join(args.output_dir, "parameters.cfg")
        if not os.path.exists(cfg_path):
            create_cfg_file(args.params, args.output_dir)

        # Execute Feature processing
        train(args)

    except Exception as e:
        print(f"\n💥 An error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# USAGE NOTE:
# Prediction mode - uses pre-existing training data for prediction:
#   python3 CleaveRNA.py --targets <fasta1> <fasta2> ... --params <params.csv> --feature_mode default --prediction_mode <train_csv_file> --model_name <model_name> --ML_target <target.csv> --output_dir <outdir>
#
# Train mode - generates training features from user FASTA files:
#   python3 CleaveRNA.py --train_mode <train1.fasta> <train2.fasta> ... --feature_mode default --output_dir <outdir> --model_name <model_name>
#   This mode generates only {model_name}_user_merged_num.csv for training feature extraction.
#   Note: feature_mode is always 'default' for train_mode.
#
# The output files will be in <outdir>.