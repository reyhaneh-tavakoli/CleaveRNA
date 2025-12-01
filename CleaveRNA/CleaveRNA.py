
import argparse
import os
import sys
import traceback
try:
    from .Feature import main as feature_main
except ImportError:
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
import shutil

def check_dependencies():
    """Check if required external tools are available"""
    missing_tools = []
    
    # Check for RNAplfold
    if not shutil.which('RNAplfold'):
        missing_tools.append('RNAplfold (ViennaRNA package)')
    
    # Check for IntaRNA
    if not shutil.which('IntaRNA'):
        missing_tools.append('IntaRNA')
    
    if missing_tools:
        print("\n❌ Missing required dependencies:")
        for tool in missing_tools:
            print(f"   • {tool}")
        
        print("\n📋 Installation instructions:")
        print("\n🔹 Option 1 - Install via conda (recommended):")
        print("   conda install -c bioconda viennarna intarna")
        
        print("\n🔹 Option 2 - Use existing conda environment:")
        print("   conda activate intarna-env  # or your environment with these tools")
        
        print("\n🔹 Option 3 - Install from source:")
        print("   ViennaRNA: https://www.tbi.univie.ac.at/RNA/")
        print("   IntaRNA: https://github.com/BackofenLab/IntaRNA")
        
        print("\n💡 After installation, verify with:")
        print("   RNAplfold --help")
        print("   IntaRNA --help")
        
        raise SystemExit(1)

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
    if args.target_files_prediction:
        file_factor = len(args.target_files_prediction) * 5  # 5 seconds per target file
    elif args.target_files_training:
        file_factor = len(args.target_files_training) * 8  # 8 seconds per training file
    else:
        file_factor = 5
    
    # Factor in feature mode complexity
    feature_complexity = {
        'default': 1.0,
        'target_screen': 1.5,
        'target_check': 1.2,
        'specific_query': 0.8
    }
    
    mode_factor = feature_complexity.get(args.prediction_mode, 1.0)
    
    # Factor in prediction vs training mode
    if args.training_file:
        mode_time = 20  # Prediction mode takes longer
    elif args.target_files_training:
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
    if 'ML_training_score' not in df.columns:
        raise ValueError("Training data must include a target column named 'ML_training_score'.")

    # Check class balance and balance if needed
    y_counts = df['ML_training_score'].value_counts()
    if len(y_counts) == 2 and y_counts.iloc[0] != y_counts.iloc[1]:
        # Suppressed balance detection message
        if progress_tracker:
            progress_tracker.update(1, "Balancing classes")
        min_count = y_counts.min()
        df_balanced = df.groupby('ML_training_score', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
        df = df_balanced
        # Suppressed balance success message
    elif len(y_counts) == 2:
        # Suppressed already balanced message
        pass
    else:
        print("⚠️ Warning: Only one class present in target column. SVM training may fail.")

    if progress_tracker:
        progress_tracker.update(2, "Preprocessing features")

    X_df = df.drop(columns=['ML_training_score'])
    y = df['ML_training_score']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X_df)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if progress_tracker:
        progress_tracker.update(3, f"Training SVM model for {model_name}")

    # Train SVM
    svm = SVC(C=1, gamma='scale', kernel='poly', degree=3, coef0=1.0, probability=True, random_state=42)
    svm.fit(X, y)

    # Save model
    model_file = f"{model_name}_SVM.pkl"
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
    svm = SVC(C=1, gamma='scale', kernel='poly', degree=3, coef0=1.0, probability=True, random_state=42)

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
    metrics_file = f"{model_name}_ML_metrics.csv"
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Mean': [np.mean(scores['accuracy']), np.mean(scores['precision']), np.mean(scores['recall']), np.mean(scores['f1'])],
        'Std': [np.std(scores['accuracy']), np.std(scores['precision']), np.std(scores['recall']), np.std(scores['f1'])]
    })
    metrics_df.to_csv(metrics_file, index=False)
    
    step_duration = time.time() - step_start_time
    if progress_tracker:
        progress_tracker.step_success("Cross-validation", step_duration)
    
    report_file_status(metrics_file, "ML metrics")

def train(args):
    # Check dependencies first
    print("🔍 Checking required dependencies...")
    check_dependencies()
    print("✅ All dependencies found!")
    
    # Initialize progress tracking
    estimated_time = predict_execution_time(args)
    print(f"\n🚀 Starting CleaveRNA analysis...")
    print(f"⏱️ Estimated execution time: {format_time(estimated_time)}")
    
    # Determine total steps based on mode
    if args.training_file:
        total_steps = 25  # More steps for prediction mode
    elif args.target_files_training:
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

        # Skip initial Feature.py run for target_files_training as it handles its own feature generation
        if not args.target_files_training:
            # Step 2: Run Feature.py
            step_start = time.time()
            print("🔧 Running Feature.py to generate 'all_generated_merged_num.csv'...")
            # Convert target files to absolute paths
            if args.target_files_prediction:
                abs_targets = [os.path.abspath(f) for f in args.target_files_prediction]
                targets_arg = ','.join(abs_targets)
            else:
                targets_arg = ''
            # Convert params file to absolute path
            abs_params = os.path.abspath(args.params)
            feature_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Feature.py'))
            feature_command = f"python3 {feature_script_path} --targets {targets_arg} --params {abs_params} --prediction_mode {args.prediction_mode} --output_dir {args.output_dir}"
            
            progress.update(2, "Executing Feature.py")
            try:
                result = subprocess.run(feature_command, shell=True, check=True, cwd=args.output_dir, capture_output=True, text=True)
                if result.stdout:
                    print("Feature.py output:", result.stdout)
                if result.stderr:
                    print("Feature.py stderr:", result.stderr)
                progress.step_success("Feature.py execution", time.time() - step_start)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error: Feature.py execution failed with error: {e}")
                if e.stdout:
                    print(f"Feature.py stdout: {e.stdout}")
                if e.stderr:
                    print(f"Feature.py stderr: {e.stderr}")
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

        if args.training_file:
            model_name = args.model_name
            target_file = args.training_scores
            default_merged_file = args.training_file

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
                f"{model_name}_run_1.csv",
                f"{model_name}_run_2.csv",
                f"{model_name}_run_3.csv"
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
            mean_std_file = f"{model_name}_statistics.csv"
            mean_std.to_csv(mean_std_file)
            report_file_status(mean_std_file, "Statistics")
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
            standardized_file = f"{model_name}_standardized_train.csv"
            df_standardized.to_csv(standardized_file, index=False)
            report_file_status(standardized_file, "Standardized train")
            progress.step_success("Data standardization", time.time() - step_start)

            # Step 8: Balance target data
            step_start = time.time()
            progress.update(2, "Balancing target data")
            df_target = pd.read_csv(target_file)
            np.random.seed(89273554)
            df_balanced = df_target.groupby('ML_training_score', group_keys=False).apply(
                lambda x: x.sample(df_target['ML_training_score'].value_counts().min(), random_state=89273554)
            )
            balanced_file = f"{model_name}_balanced_classification.csv"
            df_balanced.to_csv(balanced_file, index=False)
            report_file_status(balanced_file, "Balanced classification")

            # Skip generating non-balanced target and default ML test files
            progress.step_success("Target data balancing", time.time() - step_start)

            # Step 9: Merge training data using Dz_seq and seq2 columns
            step_start = time.time()
            progress.update(2, "Merging training datasets")
            
            # Merge balanced file (Dz_seq) with standardized file (seq2)
            # Search each Dz_seq row in seq2 column and merge if found
            print(f"🔄 Merging based on Dz_seq (balanced) ↔ seq2 (standardized)...")
            
            # Use left join to keep all balanced rows, then filter out unmatched ones
            df_merged_train = df_balanced.merge(
                df_standardized, 
                left_on='Dz_seq', 
                right_on='seq2', 
                how='left'
            )
            
            # Remove rows where seq2 is NaN (no match found in standardized file)
            initial_balanced_count = len(df_balanced)
            df_merged_train = df_merged_train.dropna(subset=['seq2'])
            final_count = len(df_merged_train)
            
            # Report merging statistics
            removed_count = initial_balanced_count - final_count
            if removed_count > 0:
                print(f"⚠️ Warning: Removed {removed_count} Dz_seq rows with no corresponding seq2 in standardized file")
                print(f"Merged dataset: {final_count} rows (from {initial_balanced_count} balanced rows)")
            else:
                print(f"✅ Successfully merged all {final_count} rows from balanced file with standardized file")
            
            # Clean up duplicate columns - drop Dz_seq and keep seq2 from standardized file
            if 'Dz_seq' in df_merged_train.columns:
                df_merged_train = df_merged_train.drop(columns=['Dz_seq'])
            
            # Ensure ML_training_score is preserved
            if 'ML_training_score' not in df_merged_train.columns:
                print("❌ Error: ML_training_score column was lost during merge")
                sys.exit(1)
            
            merged_train_file = f"{model_name}_ML_train.csv"
            df_merged_train.to_csv(merged_train_file, index=False)
            report_file_status(merged_train_file, "ML train")

            # Skip generating default ML test file
            progress.step_success("Dataset merging", time.time() - step_start)

            # Step 10: Prepare feature sets
            step_start = time.time()
            progress.update(1, "Preparing feature sets")
            feature_set = ['E_1', 'Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 'seedNumber_3', 'pumin1_4u', 'pumin1_4d']
            df_train = pd.read_csv(merged_train_file)
            feature_set_with_y = feature_set + ['ML_training_score']
            feature_set_file = f"{model_name}_ML_train_feature_set.csv"
            df_train[feature_set_with_y].to_csv(feature_set_file, index=False)
            report_file_status(feature_set_file, "ML train feature set")
            progress.step_success("Feature set preparation", time.time() - step_start)

            # Step 11: Standardize generated data
            step_start = time.time()
            progress.update(2, "Standardizing generated data")
            mean_std_file = f"{model_name}_statistics.csv"
            mean_std = pd.read_csv(mean_std_file, index_col=0)
            df_generated = pd.read_csv(os.path.join(args.output_dir, "all_generated_merged_num.csv"))
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
            train_and_save_svm(f"{model_name}_ML_train_feature_set.csv", model_name, "default_train_feature_set", progress)

            # Step 18: Create CS_info file first (moved from step 20)
            step_start = time.time()
            progress.update(1, "Creating CS_info file")
            df_all_generated = pd.read_csv(os.path.join(args.output_dir, "all_generated_merged_num.csv"))
            cs_dz_file = "CS_info.csv"
            if 'target_file' in df_all_generated.columns:
                df_all_generated[['id2', 'seq2', 'target_file']].to_csv(cs_dz_file, index=False)
            else:
                fasta_targets = args.target_files_prediction if isinstance(args.target_files_prediction, list) else args.target_files_prediction
                df_all_generated['target_file'] = None
                if len(fasta_targets) == len(df_all_generated):
                    df_all_generated['target_file'] = fasta_targets
                else:
                    df_all_generated['target_file'] = fasta_targets[0] if fasta_targets else None
                df_all_generated[['id2', 'seq2', 'target_file']].to_csv(cs_dz_file, index=False)
            report_file_status(cs_dz_file, "CS_info file")
            progress.step_success("CS_info file creation", time.time() - step_start)

            # Step 19: Make predictions
            step_start = time.time()
            progress.update(2, "Making predictions")
            # Suppressed prediction processing message
            pickle_file = f"{model_name}_SVM.pkl"
            test_file = "generated_ML_test_feature_set.csv"
            output_file = f"{model_name}_CleaveRNA_output.csv"
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
                        if 'ML_training_score' in df_test.columns:
                            y_true = df_test['ML_training_score'].reset_index(drop=True)
                            if len(y_true) != len(X_full):
                                y_true = y_true.iloc[:len(X_full)].reset_index(drop=True)
                        
                        Classification_score, reliability_score, decision_score, _ = predict_with_confidence(model, X_std, y_true)
                        predict_proba = model.predict_proba(X_std)[:, 1] if model.predict_proba(X_std).shape[1] >= 2 else model.predict_proba(X_std)[:, 0]
                        margin = np.abs(decision_score)
                        combined_score = reliability_score * margin
                        
                        # Read CS_info.csv to get id2, seq2, target_file mappings
                        cs_info_path = os.path.join(args.output_dir, "CS_info.csv")
                        if os.path.exists(cs_info_path):
                            df_cs_info = pd.read_csv(cs_info_path)
                            # Create mapping dictionaries
                            if len(df_cs_info) == len(df_test):
                                id2_mapping = df_cs_info['id2'].astype(str).tolist()
                                seq2_mapping = df_cs_info['seq2'].tolist() if 'seq2' in df_cs_info.columns else [None] * len(df_cs_info)
                                target_mapping = df_cs_info['target_file'].tolist() if 'target_file' in df_cs_info.columns else [None] * len(df_cs_info)
                            else:
                                # Fallback to original method if lengths don't match
                                id2_mapping = df_test['id2'].tolist() if 'id2' in df_test.columns else list(range(len(df_test)))
                                seq2_mapping = [None] * len(df_test)
                                target_mapping = [None] * len(df_test)
                        else:
                            # Fallback if CS_info.csv doesn't exist
                            id2_mapping = df_test['id2'].tolist() if 'id2' in df_test.columns else list(range(len(df_test)))
                            seq2_mapping = [None] * len(df_test)
                            target_mapping = [None] * len(df_test)
                        
                        result_df = pd.DataFrame({
                            'id2': id2_mapping,
                            'seq2': seq2_mapping,
                            'target_file': target_mapping,
                            'Classification_score': Classification_score,
                            'reliability_score': reliability_score,
                            'predict_proba': predict_proba,
                            'decision_score': decision_score,
                            'margin': margin,
                            'combined_score': combined_score
                        })
                        result_df.to_csv(output_path, index=False)
                        # Suppress success message
            
            progress.step_success("Predictions", time.time() - step_start)

            # Step 20: Process CS_info mapping (simplified since we already used CS_info.csv)
            step_start = time.time()
            progress.update(1, "Processing CS_info mapping")
            # This step is now simplified since we already extracted the data from CS_info.csv in predictions
            progress.step_success("CS_info mapping", time.time() - step_start)

            # Step 21: Final processing
            step_start = time.time()
            progress.update(2, "Final result processing")
            feature_set_predicted_path = os.path.join(args.output_dir, f"{model_name}_CleaveRNA_output.csv")
            if os.path.exists(feature_set_predicted_path):
                df_feature_set = pd.read_csv(feature_set_predicted_path)
                
                # Rename columns first
                column_rename_map = {
                    'id2': 'CS_Index',
                    'seq2': 'Dz_Seq',
                    'target_file': 'CS_Target_File'
                }
                df_feature_set = df_feature_set.rename(columns=column_rename_map)
                
                # Convert U to T in Dz_Seq column
                if 'Dz_Seq' in df_feature_set.columns:
                    df_feature_set['Dz_Seq'] = df_feature_set['Dz_Seq'].str.replace('U', 'T')
                
                # Update keep_cols to use new column names
                keep_cols = [col for col in ['CS_Index', 'Dz_Seq', 'CS_Target_File', 'Classification_score', 'reliability_score', 'decision_score', 'brier_score'] if col in df_feature_set.columns]
                df_feature_set = df_feature_set[keep_cols]
                
                # Remove any duplicate columns
                cols_to_remove = [col for col in df_feature_set.columns if col in ['CS_Index.1', 'Dz_Seq.1', 'CS_Target_File.1'] or col.startswith('CS_Index.') or col.startswith('Dz_Seq.') or col.startswith('CS_Target_File.')]
                if cols_to_remove:
                    df_feature_set = df_feature_set.drop(columns=cols_to_remove)
                
                df_feature_set.to_csv(feature_set_predicted_path, index=False)
                report_file_status(feature_set_predicted_path, "Updated CleaveRNA output")
                
                df_feature_set = pd.read_csv(feature_set_predicted_path)
                df_feature_set = df_feature_set.sort_values(by=['Classification_score', 'reliability_score'], ascending=[False, False])
                df_feature_set.to_csv(feature_set_predicted_path, index=False)
                report_file_status(feature_set_predicted_path, "Sorted CleaveRNA output")
            progress.step_success("Final result processing", time.time() - step_start)

        elif args.target_files_training:
            model_name = args.model_name
            user_train_files = args.target_files_training
            
            # Step 4: Generate user features
            step_start = time.time()
            progress.update(3, "Generating user training features")
            feature_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Feature.py'))
            user_merged_file = os.path.join(args.output_dir, f"{model_name}_user_merged_num.csv")
            # Convert target files and params to absolute paths
            abs_user_train_files = [os.path.abspath(f) for f in user_train_files]
            abs_params = os.path.abspath(args.params)
            user_train_command = f"python3 {feature_script_path} --targets {','.join(abs_user_train_files)} --params {abs_params} --prediction_mode default --output_dir {args.output_dir}"
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
            print("❌ Error: Either --training_file or --target_files_training must be provided.")
            sys.exit(1)

        # Cleanup intermediate files
        step_start = time.time()
        print("\n🧹 Cleaning up intermediate files...")
        model_name = args.model_name
        
        # Files to keep (essential outputs)
        keep_files = {
            f"{model_name}_ML_metrics.csv",
            f"{model_name}_CleaveRNA_output.csv", 
            f"{model_name}_ML_train.csv",
            f"{model_name}_ML_train_feature_set.csv",
            f"{model_name}_balanced_classification.csv",
            "parameters.cfg"
        }
        
        # Add training mode specific files to keep
        if args.target_files_training:
            keep_files.add(f"{model_name}_user_merged_num.csv")
        
        # Always keep user_merged_num.csv file at the end
        keep_files.add(f"{model_name}_user_merged_num.csv")
        
        # Input files to protect (never remove these)
        protected_input_files = set()
        
        # Protect user-provided files
        if args.params:
            protected_input_files.add(os.path.basename(args.params))
        if args.training_scores:
            protected_input_files.add(os.path.basename(args.training_scores))
        if args.specific_query_input:
            protected_input_files.add(os.path.basename(args.specific_query_input))
        if args.prediction_mode:
            protected_input_files.add(os.path.basename(args.prediction_mode))
        
        # Protect FASTA input files
        if args.target_files_prediction:
            for target in args.target_files_prediction:
                protected_input_files.add(os.path.basename(target))
        if args.target_files_training:
            for train_file in args.target_files_training:
                protected_input_files.add(os.path.basename(train_file))
        
        # List of all potential intermediate files to remove
        cleanup_files = [
            "all_generated_merged_num.csv",
            "standardized_all_generated_merged_num.csv",
            "generated_ML_test_feature_set.csv",
            "CS_info.csv",
            f"{model_name}_statistics.csv",
            f"{model_name}_standardized_train.csv",
            f"{model_name}_ML_train_feature_set.csv",
            f"{model_name}_SVM.pkl",
            f"{model_name}_user_merged_num.csv",
            f"{model_name}_Results_with_region.csv",
            f"{model_name}_Results_without_region.csv",
            f"{model_name}_Results_pairwise.csv"
        ]
        
        # Remove files that exist and are not in keep_files
        removed_count = 0
        
        # First, remove files from the static cleanup list
        for file_to_remove in cleanup_files:
            file_path = os.path.join(args.output_dir, file_to_remove)
            file_name = os.path.basename(file_to_remove)
            if (os.path.exists(file_path) and 
                file_to_remove not in keep_files and 
                file_name not in protected_input_files):
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Warning: Could not remove {file_to_remove}: {e}")
        
        # Then, scan directory for target-specific Result files
        import glob
        result_patterns = [
            "*_Results_with_region.csv",
            "*_Results_without_region.csv", 
            "*_Results_pairwise.csv"
        ]
        
        for pattern in result_patterns:
            pattern_path = os.path.join(args.output_dir, pattern)
            matching_files = glob.glob(pattern_path)
            for file_path in matching_files:
                file_name = os.path.basename(file_path)
                if (file_name not in keep_files and 
                    file_name not in protected_input_files):
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"⚠️ Warning: Could not remove {file_name}: {e}")
        
        # Clean up generated_merged_num.csv files
        generated_merged_pattern = "generated_merged_num.csv"
        generated_merged_files = glob.glob(generated_merged_pattern)
        for file_path in generated_merged_files:
            file_name = os.path.basename(file_path)
            if (os.path.isfile(file_path) and 
                file_name not in protected_input_files):
                try:
                    os.remove(file_path)
                    removed_count += 1
                    print(f"📄 Removed generated merged file: {file_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not remove file {file_path}: {e}")
        
        # Finally, clean up RNAplfold output directories
        import shutil
        rnaplfold_pattern = "rnaplfold_output_*"
        rnaplfold_dirs = glob.glob(rnaplfold_pattern)
        for dir_path in rnaplfold_dirs:
            if os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    removed_count += 1
                    print(f"🗂️ Removed RNAplfold directory: {dir_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not remove directory {dir_path}: {e}")
        
        cleanup_duration = time.time() - step_start
        if removed_count > 0:
            print(f"✅ Cleaned up {removed_count} intermediate files in {cleanup_duration:.2f}s")
        else:
            print(f"✅ No intermediate files to clean up")

        # Final completion
        progress.update(progress.total_steps - progress.current_step, "Finalizing")
        total_time = time.time() - progress.start_time
        progress.finish()
        
        print(f"\n🎊 CleaveRNA analysis completed successfully!")
        print(f"⏱️ Total execution time: {format_time(total_time)}")
        print(f"📊 Final output files available in: {args.output_dir}")
        print(f"📋 Kept files: {', '.join(sorted(keep_files))}")
        
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
        
        parser = argparse.ArgumentParser(
            prog='CleaveRNA',
            description="""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        🧬 CleaveRNA Analysis Tool 🧬                        │
│                                                                             │
│  Advanced machine learning-based computational tool for scoring candidate   │
│  DNAzyme cleavage sites in substrate RNA sequences using structural and     │
│  thermodynamic features.                                                    │
│                                                                             │
│  Features:                                                                  │
│  • Machine Learning Prediction Models                                       │
│  • RNA Secondary Structure Analysis                                         │
│  • Thermodynamic Feature Extraction                                         │
│  • Multiple Prediction Modes                                                │
│  • Cross-validation and Performance Metrics                                 │
└─────────────────────────────────────────────────────────────────────────────┘
            """,
            epilog="""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📋 USAGE EXAMPLES                              │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 TRAINING MODE - Generate features for model training:
┌─────────────────────────────────────────────────────────────────────────────┐
│ cleaverna --target_files_training train1.fasta train2.fasta \\              │
│           --params parameters.csv --prediction_mode default \\              │
│           --model_name my_model --output_dir results/                       │
└─────────────────────────────────────────────────────────────────────────────┘

🔮 PREDICTION MODE - Standard cleavage site prediction:
┌─────────────────────────────────────────────────────────────────────────────┐
│ cleaverna --target_files_prediction seq1.fasta seq2.fasta \\                │
│           --params parameters.csv --prediction_mode default \\              │
│           --training_file training_data.csv \\                              │
│           --training_scores labels.csv \\                                   │
│           --model_name prediction_model --output_dir results/               │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 TARGET SCREENING - Screen specific cleavage sites:
┌─────────────────────────────────────────────────────────────────────────────┐
│ cleaverna --target_files_prediction target.fasta \\                         │
│           --params screen_params.csv --prediction_mode target_screen \\     │
│           --training_file training_data.csv \\                              │
│           --training_scores labels.csv \\                                   │
│           --model_name screen_model --output_dir screening_results/         │
└─────────────────────────────────────────────────────────────────────────────┘

🔍 TARGET CHECK - Validate cleavage sites in specific regions:
┌─────────────────────────────────────────────────────────────────────────────┐
│ cleaverna --target_files_prediction target.fasta \\                         │
│           --params check_params.csv --prediction_mode target_check \\       │
│           --training_file training_data.csv \\                              │
│           --training_scores labels.csv \\                                   │
│           --model_name check_model --output_dir check_results/              │
└─────────────────────────────────────────────────────────────────────────────┘

⚡ SPECIFIC QUERY - Custom DNAzyme sequence analysis:
┌─────────────────────────────────────────────────────────────────────────────┐
│ cleaverna --target_files_prediction target.fasta \\                         │
│           --params query_params.csv --prediction_mode specific_query \\     │
│           --specific_query_input custom_queries.csv \\                      │
│           --training_file training_data.csv \\                              │
│           --training_scores labels.csv \\                                   │
│           --model_name query_model --output_dir query_results/              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            🔗 MORE INFORMATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

📚 Documentation: https://github.com/reyhaneh-tavakoli/CleaveRNA
💡 Issues & Support: https://github.com/reyhaneh-tavakoli/CleaveRNA/issues

            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False
        )
        
        # Add custom help argument with improved formatting
        parser.add_argument(
            '-h', '--help',
            action='help',
            help='Show this help message and exit'
        )
        
        # Required Arguments Group
        required_group = parser.add_argument_group('🔴 REQUIRED ARGUMENTS')
        required_group.add_argument(
            '--model_name', 
            required=True, 
            metavar='NAME',
            help='Model identifier and output file prefix (e.g., "my_experiment")'
        )
        
        # Input Files Group  
        input_group = parser.add_argument_group('📁 INPUT FILES')
        input_group.add_argument(
            '--target_files_prediction', 
            nargs='+', 
            metavar='FASTA',
            help='Target RNA sequence files in FASTA format (required for prediction mode)'
        )
        input_group.add_argument(
            '--target_files_training', 
            nargs='+', 
            metavar='FASTA',
            help='Training RNA sequence files in FASTA format (required for training mode)'
        )
        input_group.add_argument(
            '--params', 
            metavar='CSV',
            help='Parameters file with columns: LA, RA, CS, Tem, CA'
        )
        input_group.add_argument(
            '--training_file', 
            metavar='CSV',
            help='Training data feature matrix (mutually exclusive with --target_files_training)'
        )
        input_group.add_argument(
            '--training_scores', 
            metavar='CSV',
            help='Training target labels/scores (required with --training_file)'
        )
        input_group.add_argument(
            '--specific_query_input', 
            metavar='CSV',
            help='Custom query parameters for specific_query mode',
            default=None
        )
        
        # Analysis Options Group
        analysis_group = parser.add_argument_group('🔬 ANALYSIS OPTIONS')
        analysis_group.add_argument(
            '--prediction_mode', 
            choices=['default', 'target_screen', 'target_check', 'specific_query'],
            metavar='MODE',
            help='Analysis mode selection:\n'
                 '• default : Standard cleavage site prediction\n'
                 '• target_screen : Screen custom cleavage sites\n'
                 '• target_check : Validate sites in specific regions\n'
                 '• specific_query : Analyze custom DNAzyme sequences'
        )
        
        # Output Options Group
        output_group = parser.add_argument_group('📤 OUTPUT OPTIONS') 
        output_group.add_argument(
            '--output_dir', 
            metavar='DIR',
            help='Output directory for results (default: current directory)', 
            default='.'
        )
        args = parser.parse_args()

        # Show initial configuration
        print(f"🔧 Configuration loaded:")
        print(f"   Model name: {args.model_name}")
        print(f"   Prediction mode: {args.prediction_mode}")
        if args.training_file:
            print(f"   Mode: Prediction")
            print(f"   Targets: {len(args.target_files_prediction) if args.target_files_prediction else 0} files")
        elif args.target_files_training:
            print(f"   Mode: Training")
            print(f"   Training files: {len(args.target_files_training)} files")
        print(f"   Output directory: {args.output_dir}")

        # Validate that one of the train modes is specified
        if not args.training_file and not args.target_files_training:
            print("❌ Error: Either --training_file or --target_files_training must be provided.")
            sys.exit(1)

        # Validate ML_training_score is provided for training_file mode
        if args.training_file and not args.training_scores:
            print("❌ Error: --training_scores is required when using --training_file.")
            sys.exit(1)

        # Validate targets is provided for training_file mode
        if args.training_file and not args.target_files_prediction:
            print("❌ Error: --target_files_prediction is required when using --training_file.")
            sys.exit(1)

        # Validate each target file in the list (only for training_file mode)
        if args.target_files_prediction:
            for target in args.target_files_prediction:
                if not os.path.exists(target):
                    print(f"❌ Error: Target file '{target}' does not exist.")
                    sys.exit(1)

        if not os.path.exists(args.params):
            print(f"❌ Error: Parameters file '{args.params}' does not exist.")
            sys.exit(1)

        if args.prediction_mode == 'specific_query' and args.specific_query_input and not os.path.exists(args.specific_query_input):
            print(f"❌ Error: Specific CSV file '{args.specific_query_input}' does not exist.")
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
#   python3 CleaveRNA.py --target_files_prediction <fasta1> <fasta2> ... --params <params.csv> --prediction_mode default --training_file <train_csv_file> --model_name <model_name> --training_scores <target.csv> --output_dir <outdir>
#
# Train mode - generates training features from user FASTA files:
#   python3 CleaveRNA.py --target_files_training <train1.fasta> <train2.fasta> ... --prediction_mode default --output_dir <outdir> --model_name <model_name>
#   This mode generates only {model_name}_user_merged_num.csv for training feature extraction.
#   Note: prediction_mode is always 'default' for target_files_training.
#
# The output files will be in <outdir>.