#!/usr/bin/env python3

import sys
import subprocess
import re
import time
import os
import pandas as pd
import argparse
from datetime import datetime
import glob
import numpy as np
import contextlib

def convert_t_to_u(sequence):
    return sequence.replace('T', 'U')

def run_rnaplfold(target_file, LA, RA, output_dir, temperature):
    os.makedirs(output_dir, exist_ok=True)  # Create a unique directory for each file
    u_length = LA + RA + 4
    target_file_path = os.path.abspath(target_file)
    command = f"RNAplfold -W 150 -L 100 -u {u_length} -T {temperature} < \"{target_file_path}\""
    print(f"\nExecuting RNAplfold command: {command} in {output_dir}")
    
    try:
        subprocess.run(command, shell=True, check=True, cwd=output_dir)
        lunp_files = [f for f in os.listdir(output_dir) if f.endswith('_lunp')]
        if not lunp_files:
            raise FileNotFoundError("RNAplfold did not generate a _lunp file.")
        return os.path.join(output_dir, lunp_files[0])
    except subprocess.CalledProcessError as e:
        print(f"RNAplfold failed:\n{e.stderr}")
        raise

def parse_rnaplfold_output(output_file):
    print(f"\nParsing RNAplfold output: {output_file}")
    unpaired_probs = {}
    with open(output_file, "r") as file:
        for line in file:
            if line.startswith("#") or line.startswith("i$"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    unpaired_probs[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    if not unpaired_probs:
        raise ValueError("No valid data found in RNAplfold output.")
    return unpaired_probs

def find_CS(sequence, CS_list):
    matches, used_positions = [], set()
    for motif in CS_list:
        for match in re.finditer(motif, sequence):
            start = match.start()
            pos1, pos2 = start + 1, start + 2
            key = f"{pos1}-{pos2}"
            if key not in used_positions:
                used_positions.add(key)
                matches.append((max(0, start - 10), min(len(sequence), match.end() + 10), match.group(), pos1, pos2))
    return matches

def prepare_sequences(sequence, CS, left_arm_length, right_arm_length, core):
    valid = {"A", "U", "C", "G"}
    queries = []
    for start, end, motif, pos1, pos2 in CS:
        if pos1 <= 8 or pos2 <= 8:
            continue
        left_arm = sequence[pos1:pos1 + left_arm_length]
        right_arm = sequence[pos2 - right_arm_length - 3:pos2 - 2]
        if not set(left_arm).issubset(valid) or not set(right_arm).issubset(valid):
            continue
        comp = lambda s: ''.join(["AUCG"["UAGC".index(n)] for n in s][::-1])
        query_seq = f"{comp(left_arm)}{core}{comp(right_arm)}"
        queries.append((f"{motif}-{pos1}-{pos2}", query_seq))
    return queries

def write_queries_to_fasta(queries, query_file):
    with open(query_file, "w") as f:
        for name, seq in queries:
            f.write(f">{name}\n{seq}\n")

def process_specific_query(csv_file, output_dir):
    """
    Process specific_query mode by validating the CS motif and generating query files.
    """
    df = pd.read_csv(csv_file)
    if not {'LA_seq', 'RA_seq', 'CS', 'CS_Index_query', 'Tem', 'CA'}.issubset(df.columns):
        raise ValueError("The CSV file for specific_query mode must contain 'LA_seq', 'RA_seq', 'CS', 'CS_Index_query', 'Tem', and 'CA' columns.")

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for _, row in df.iterrows():
        LA_seq = row['LA_seq']
        RA_seq = row['RA_seq']
        CS = row['CS']
        CA = row['CA']
        temperature = float(row['Tem'])
        target_file, position = row['CS_Index_query'].split(':')
        start, end = map(int, position.split('-'))

        with open(target_file, 'r') as f:
            target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

        # Validate the CS motif
        sequence_at_position = target_seq[start - 1:end]  # Adjust for 0-based indexing
        if sequence_at_position.upper() != CS.upper():
            print(f"Warning: CS motif '{CS}' does not match the sequence at position {start}-{end} in {target_file}. Found: '{sequence_at_position}'. Using the sequence from the target file.")
            CS = sequence_at_position.upper()

        # Generate the query sequence using LA_seq and RA_seq directly (no reverse complement)
        query_seq = f"{LA_seq}{CA}{RA_seq}"

        # Write the query to a FASTA file
        query_file = os.path.join(output_dir, f"queries_{os.path.basename(target_file).split('.')[0]}_{start}_{end}.fasta")
        with open(query_file, "w") as f:
            f.write(f">{CS}-{start}-{end}\n{query_seq}\n")

        print(f"Generated query file: {query_file}")

        # Run RNAplfold and process IntaRNA queries
        output_prefix = os.path.basename(target_file).split('.')[0]
        output_dir = f"rnaplfold_output_{output_prefix}"
        os.makedirs(output_dir, exist_ok=True)
        
        lunp_file = run_rnaplfold(target_file, len(LA_seq), len(RA_seq), output_dir, temperature)
        
        # Use parameters.cfg for IntaRNA, not the csv file
        param_file = os.path.join(os.path.dirname(__file__), 'parameters.cfg')
        process_intarna_queries(target_file, query_file, lunp_file, param_file, len(LA_seq), len(RA_seq), output_prefix, CS)
        
        # Post-process the features
        post_process_features(target_file, output_dir)

def construct_intarna_command(query_file, target_file, param_file, additional):
    return (
        f"IntaRNA --query {query_file} --target {target_file} "
        f"--qIntLenMax 0 --parameterFile {param_file} "
        f"--outMode C --outNumber 1 --outOverlap N "
        f"--outCsvCols 'id2,seq2,E,ED1,ED2,Pu1,Pu2,subseqDB,hybridDB,E_hybrid,"
        f"seedStart1,seedEnd1,seedStart2,seedEnd2,seedE,seedED1,seedED2,seedPu1,seedPu2,P_E' "
        + additional
    )

def process_intarna_queries(target_file, query_file, lunp_file, param_file, left_arm, right_arm, output_prefix, cs_name):
    with open(target_file, "r") as f:
        sequence = ''.join([line.strip() for line in f if not line.startswith(">")])
    queries, all_results = [], []
    with open(query_file, "r") as f:
        name, seq = "", ""
        for line in f:
            if line.startswith(">"):
                if name and seq:
                    queries.append((name, seq))
                name = line.strip()[1:]
                seq = ""
            else:
                seq += line.strip()
        if name and seq:
            queries.append((name, seq))

    for i, (name, seq) in enumerate(queries, 1):
        tmp_q = f"temp_query_{i}.fasta"
        with open(tmp_q, "w") as f:
            f.write(f">{name}\n{seq}\n")
        parts = name.split("-")
        if len(parts) >= 3:
            start, end = max(1, int(parts[1]) - 9), min(len(sequence), int(parts[2]) + 11)
            total_len = left_arm + right_arm + 3
            region = f"{start}-{end}"
            cmds = [
                construct_intarna_command(tmp_q, target_file, param_file,
                    f"--tAcc P --tIntLenMax {total_len} --tAccFile {lunp_file} --tRegion {region} --out {output_prefix}_result_{i}_with_region.csv"),
                construct_intarna_command(tmp_q, target_file, param_file,
                    f"--tAcc P --tIntLenMax {total_len} --tAccFile {lunp_file} --out {output_prefix}_result_{i}_without_region.csv"),
                construct_intarna_command(tmp_q, tmp_q, param_file,
                    f"--out {output_prefix}_result_{i}_pairwise.csv")
            ]
            for cmd in cmds:
                try:
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error executing IntaRNA command: {cmd}\n{e}")
                    continue

            # Check if the expected files were created
            expected_files = [
                f"{output_prefix}_result_{i}_with_region.csv",
                f"{output_prefix}_result_{i}_without_region.csv",
                f"{output_prefix}_result_{i}_pairwise.csv"
            ]
            for file in expected_files:
                if os.path.exists(file):
                    all_results.append(file)
                else:
                    print(f"Warning: Expected file {file} was not created.")
        os.remove(tmp_q)

    # Correcting the file naming logic to use the correct file names
    final_files = [
        f"{output_prefix}_Results_with_region.csv",
        f"{output_prefix}_Results_without_region.csv",
        f"{output_prefix}_Results_pairwise.csv"
    ]

    for out, res_group in zip(final_files, [all_results[i::3] for i in range(3)]):
        with open(out, "w") as outf:
            if res_group:
                with open(res_group[0]) as first: outf.write(first.readline())
                for r in res_group:
                    with open(r) as f: next(f); outf.write(f.read())
    for f in os.listdir():
        if f.startswith(f"{output_prefix}_result_") and f.endswith(".csv"):
            os.remove(f)

    # Collect all result files for all targets
    all_result_files = []
    for file in os.listdir():
        if file.endswith("_Results_with_region.csv") or \
           file.endswith("_Results_without_region.csv") or \
           file.endswith("_Results_pairwise.csv"):
            all_result_files.append(file)

    # Extract unique prefixes from the result files
    unique_prefixes = list(set(file.rsplit("_Results", 1)[0] for file in all_result_files))

    # Correctly define the result files for each prefix
    for prefix in unique_prefixes:
        result_files = {
            "with_region": f"{prefix}_Results_with_region.csv",
            "without_region": f"{prefix}_Results_without_region.csv",
            "pairwise": f"{prefix}_Results_pairwise.csv"
        }

        print(f"Processing prefix: {prefix}")

        for key, file in result_files.items():
            if not os.path.exists(file):
                print(f"File not found: {file}")
            else:
                print(f"Found file: {file}")

    # Merge numerical columns for all unique prefixes
    merge_numerical_columns(output_file=None, output_prefixes=unique_prefixes)

def merge_numerical_columns(output_file=None, output_prefixes=None):
    """
    Merge numerical columns from multiple result files for all provided prefixes.
    """
    if output_prefixes is None or not isinstance(output_prefixes, list):
        print("No output prefixes provided for merge_numerical_columns, using default")
        output_prefixes = ["5"]  # This was the hardcoded default

    print(f"\nMerging numerical columns for prefixes: {', '.join(output_prefixes)}...")

    merged_data = pd.DataFrame()

    for output_prefix in output_prefixes:
        print(f"Processing prefix: {output_prefix}")

        # Define the specific files to merge based on the output_prefix
        result_files = [
            f"{output_prefix}_Results_with_region.csv",
            f"{output_prefix}_Results_without_region.csv",
            f"{output_prefix}_Results_pairwise.csv"
        ]

        for idx, file in enumerate(result_files, start=1):
            if os.path.exists(file):
                print(f"Processing file: {file}")  # Debug statement
                try:
                    if os.stat(file).st_size == 0:  # Check if the file is empty
                        print(f"Warning: File {file} is empty. Skipping.")
                        continue

                    data = pd.read_csv(file)
                    print(f"File content preview:\n{data.head()}\n")  # Debug statement

                    if data.empty:
                        print(f"Warning: File {file} contains no data. Skipping.")
                        continue

                    numerical_cols = data.select_dtypes(include=['number'])
                    numerical_cols = numerical_cols.add_suffix(f"_{output_prefix}_{idx}")

                    if 'id2' in data.columns:
                        numerical_cols.insert(0, 'id2', data['id2'])

                    merged_data = pd.concat([merged_data, numerical_cols], axis=0, ignore_index=True)
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
            else:
                print(f"File not found: {file}")

    if not merged_data.empty:
        print(f"✓ Merged numerical columns processed successfully.")
        print(f"Merged data preview:\n{merged_data.head()}\n")
    else:
        print("✗ No numerical columns found to merge.")

    # Ensure 'id2' is a single column before inserting
    if 'id2' in merged_data.columns:
        numeric_columns = merged_data.select_dtypes(include=['number'])
        numeric_columns.insert(0, 'id2', merged_data['id2'])
        if 'seq2' in merged_data.columns:
            numeric_columns.insert(1, 'seq2', merged_data['seq2'])
        else:
            print("Warning: 'seq2' is missing in the merged data.")
    else:
        print("Warning: 'id2' is missing in the merged data.")
        numeric_columns = merged_data.select_dtypes(include=['number'])

    if output_file is None:
        output_file = "generated_merged_num.csv"

    numeric_columns.to_csv(output_file, index=False)
    print(f"✓ Saved id2, seq2, and numerical columns to {output_file}")
    print("Numerical data sample:\n", numeric_columns.head())

def post_process_features(target_file, output_dir):
    """Perform additional feature processing after Feature.py completes"""
    print("\nStarting post-processing...")
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
        print("Unpaired data sample:\n", pu.head())

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
            print(f"Checking existence of result file: {result_file}")  # Added logging
            if not os.path.exists(result_file):
                print(f"✗ Result file {result_file} not found.")  # Added logging
                raise FileNotFoundError(f"✗ Result file {result_file} not found. Did IntaRNA run correctly?")

            print(f"✓ Processing {result_file}...")
            df = pd.read_csv(result_file)
            print(f"Result file {i} sample:\n{df.head()}")

            # Verify required columns exist
            required_cols = ['id2', 'seq2', 'seedE']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"✗ Missing required columns in {result_file}: {missing_cols}")  # Added logging
                raise ValueError(f"✗ Missing required columns in {result_file}: {missing_cols}")

            # Ensure 'seedE' column is treated as a string before using .str accessor
            if 'seedE' in df.columns:
                df['seedE'] = df['seedE'].astype(str)

            # Number of seeds
            df['seedNumber'] = df['seedE'].str.count(':') + 1
            df['seedEbest'] = df['seedE'].str.extract(r'^([^:]+)').astype(float)

            # Add suffix to column names
            df.columns = [f"{col}_{i}" for col in df.columns]
            out.append(df)

        print("All result files processed successfully.")  # Added logging

        print("Merging datasets...")
        merged_data = out[0].merge(out[1], 
                                 left_on=["id2_1", "seq2_1"],
                                 right_on=["id2_2", "seq2_2"],
                                 how='outer')
        merged_data = merged_data.merge(out[2], 
                                      left_on=["id2_1", "seq2_1"],
                                      right_on=["id2_3", "seq2_3"],
                                      how='outer')
        merged_data = merged_data.rename(columns={'id2_1': 'id2', 'seq2_1': 'seq2'})
        print("Merged data columns:", merged_data.columns.tolist())
        print("Merged data sample:\n", merged_data.head())

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
                    pos = int(row['pos']) - 1  # Adjust for 0-based indexing in pandas
                    if direction == 'u':
                        return min(pu.loc[pos - p, 'l1'] for p in positions if (pos - p) >= 0)
                    else:
                        return min(pu.loc[pos + p, 'l1'] for p in positions if (pos + p) < len(pu))
                except (ValueError, KeyError):
                    return float('nan')

            for name, positions, direction in [
                ('pumin1_4u', [1, 2, 3, 4], 'u'),
                ('pumin5_8u', [5, 6, 7, 8], 'u'),
                ('pumin1_4d', [1, 2, 3, 4], 'd'),
                ('pumin5_8d', [5, 6, 7, 8], 'd')
            ]:
                merged_data[name] = merged_data.apply(
                    lambda x: safe_get_min_pu(x, positions, direction), axis=1)

            merged_data = merged_data.drop(columns=['pos'])

        os.makedirs(output_dir, exist_ok=True)

        full_output_path = os.path.join(output_dir, f"{base_filename}_generated_merged.csv")
        merged_data.to_csv(full_output_path, index=False)
        print(f"✓ Saved full merged data to {full_output_path}")
        print("Full data sample:\n", merged_data.head())

        numeric_columns = merged_data.select_dtypes(include=['number'])

        if 'id2' in merged_data.columns:
            numeric_with_id = pd.DataFrame()
            numeric_with_id['id2'] = merged_data['id2']

            if 'seq2' in merged_data.columns:
                numeric_with_id['seq2'] = merged_data['seq2']

            for col in numeric_columns.columns:
                numeric_with_id[col] = numeric_columns[col]

            num_output_path = os.path.join(output_dir, "generated_merged_num.csv")
            numeric_with_id.to_csv(num_output_path, index=False)
            print(f"✓ Saved id2, seq2, and numerical columns to {num_output_path}")
            print("Numerical data sample:\n", numeric_with_id.head())
        else:
            print("✗ Warning: 'id2' column not found in merged data")

        print(f"Checking if generated_merged_num.csv exists in {output_dir}...")
        if os.path.exists(os.path.join(output_dir, "generated_merged_num.csv")):
            print(f"✓ File generated_merged_num.csv found in {output_dir}")
        else:
            print(f"✗ File generated_merged_num.csv not found in {output_dir}")
            raise FileNotFoundError("generated_merged_num.csv was not created.")
    except Exception as e:
        print(f"✗ Post-processing failed: {e}")

def merge_all_generated_files(output_dir, final_output_file, targets_fasta_files=None):
    """
    Merge all generated_merged_num.csv files from the output directories into one file.
    Add a column with the target fasta file name for each id2 by searching which file each id2 belongs to.
    If targets_fasta_files is provided, only include those files.
    """
    print("\nMerging all generated_merged_num.csv files for target_screen mode...")

    # Look for rnaplfold_output_* directories in the working directory
    relevant_directories = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("rnaplfold_output_")
    ]

    # If targets_fasta_files is provided, filter relevant_directories to only those
    if targets_fasta_files is not None:
        target_basenames = set([os.path.splitext(os.path.basename(f))[0] for f in targets_fasta_files])
        relevant_directories = [d for d in relevant_directories if d.replace("rnaplfold_output_", "") in target_basenames]

    merged_data = pd.DataFrame()
    for directory in relevant_directories:
        # Extract the actual FASTA file name from the directory name
        fasta_name = directory.replace("rnaplfold_output_", "")
        fasta_file = fasta_name if fasta_name.endswith(".fasta") else f"{fasta_name}.fasta"
        fasta_file = os.path.basename(fasta_file)  # Only the filename
        for root, _, files in os.walk(directory):
            for file in files:
                if file == "generated_merged_num.csv":
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    # Always set target_file column to the FASTA file name for every row
                    df['target_file'] = fasta_file
                    merged_data = pd.concat([merged_data, df], ignore_index=True)

    if merged_data.empty:
        print("✗ No data to save after merging.")
        return

    # Save the merged data to the final output file
    # Remove any duplicate 'id2.1', 'seq2.1', 'target_file.1' and similar columns before saving
    cols_to_remove = [col for col in merged_data.columns if col in ['id2.1', 'seq2.1', 'target_file.1'] or 
                      col.startswith('id2.') or col.startswith('seq2.') or col.startswith('target_file.')]
    if cols_to_remove:
        merged_data = merged_data.drop(columns=cols_to_remove)
    merged_data.to_csv(final_output_file, index=False)
    print(f"✓ Merged data saved to {final_output_file} (with only one set of id2, seq2, target_file columns)")

@contextlib.contextmanager
def change_working_dir(new_dir):
    prev_dir = os.getcwd()
    os.makedirs(new_dir, exist_ok=True)
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--targets', required=True, help="Path to a directory or a comma-separated list of FASTA files")
        parser.add_argument('--params', required=True, help="Path to the CSV file containing parameters")
        parser.add_argument('--feature_mode', required=True, choices=['default', 'target_screen', 'target_check', 'specific_query'], help="Mode of operation")
        parser.add_argument('--output_dir', required=False, default='.', help="Directory to save all outputs")
        args = parser.parse_args()
        
        # Make params file also serve as specific_csv for specific_query mode
        if args.feature_mode == 'specific_query':
            args.specific_csv = args.params

    params_df = pd.read_csv(args.params)

    # Initialize required_columns to avoid UnboundLocalError
    required_columns = set()

    if args.feature_mode == 'specific_query':
        required_columns = {'LA_seq', 'RA_seq', 'CS', 'CS_Index_query', 'Tem', 'CA'}
        missing_columns = required_columns - set(params_df.columns)
        if missing_columns:
            raise ValueError(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")

    elif args.feature_mode == 'default':
        required_columns = {'LA', 'RA', 'CS', 'Tem', 'CA'}
        if not required_columns.issubset(params_df.columns):
            raise ValueError("The CSV file must contain the columns: LA, RA, CS, Tem, and CA.")

    elif args.feature_mode == 'target_screen':
        required_columns = {'LA', 'RA', 'CS', 'CS_index', 'Tem', 'CA'}
        missing_columns = required_columns - set(params_df.columns)
        if missing_columns:
            raise ValueError(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")

    elif args.feature_mode == 'target_check':
        required_columns = {'LA', 'RA', 'CS', 'Start_End_Index', 'Tem', 'CA'}
        missing_columns = required_columns - set(params_df.columns)
        if missing_columns:
            raise ValueError(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")

    missing_columns = required_columns - set(params_df.columns)
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")

    if '' in params_df.columns:
        raise ValueError("The CSV file contains unnamed or empty column headers. Please fix the file.")

    params_df = params_df.loc[:, ~params_df.columns.str.contains('^Unnamed')]

    if args.feature_mode == 'default':
        if not {'LA', 'RA', 'CS', 'Tem', 'CA'}.issubset(params_df.columns):
            raise ValueError("The CSV file must contain the columns: LA, RA, CS, Tem, and CA.")

        for _, row in params_df.iterrows():
            LA, RA, CS, temperature, core = int(row['LA']), int(row['RA']), row['CS'].split(','), float(row['Tem']), row['CA']
            for fasta_file in args.targets.split(','):
                with open(fasta_file, 'r') as f:
                    target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

                for motif in CS:
                    motif_matches = find_CS(target_seq, [motif])
                    queries = prepare_sequences(target_seq, motif_matches, LA, RA, core)
                    query_file = f"queries_{os.path.basename(fasta_file).split('.')[0]}_{motif}.fasta"
                    write_queries_to_fasta(queries, query_file)

    elif args.feature_mode == 'target_screen':
        if not {'LA', 'RA', 'CS', 'CS_index', 'Tem', 'CA'}.issubset(params_df.columns):
            raise ValueError("The CSV file must contain the columns: LA, RA, CS, CS_index, Tem, and CA.")

        processed_files = set()
        cs_dz_records = []  # To store id2, seq2, target_file for each query in order
        for _, row in params_df.iterrows():
            LA, RA, CS, temperature, core = int(row['LA']), int(row['RA']), row['CS'].split(','), float(row['Tem']), row['CA']
            target_file, region = row['CS_index'].split(':')
            processed_files.add(target_file)
            start, end = map(int, region.split('-'))

            with open(target_file, 'r') as f:
                target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

            # Always extract a dinucleotide: positions start and start+1 (1-based)
            dinuc_start = start
            dinuc_end = start + 1
            motif = target_seq[dinuc_start - 1:dinuc_end]  # 1-based inclusive
            if len(motif) != 2:
                print(f"Warning: Dinucleotide extraction failed at {dinuc_start}-{dinuc_end} in {target_file}. Extracted: '{motif}'. Skipping.")
                continue
            if (end - start + 1) != 2:
                print(f"Warning: Provided region {start}-{end} is not a dinucleotide. Using dinucleotide at {dinuc_start}-{dinuc_end} instead.")

            # Format id2 as {dinucleotide}-{start}-{end}, e.g., AC-17-18
            id2 = f"{motif}-{dinuc_start}-{dinuc_end}"
            motif_matches = [(dinuc_start, dinuc_end, motif, dinuc_start, dinuc_end)]
            queries = prepare_sequences(target_seq, motif_matches, LA, RA, core)
            # Overwrite the id2 in queries with the new format
            queries = [(id2, seq) for _, seq in queries]
            query_file = f"queries_{os.path.basename(target_file).split('.')[0]}.fasta"
            write_queries_to_fasta(queries, query_file)
            print(f"Generated query file: {query_file}")

            output_prefix = os.path.basename(target_file).split('.')[0]
            output_dir = f"rnaplfold_output_{output_prefix}"
            os.makedirs(output_dir, exist_ok=True)
            lunp_file = run_rnaplfold(target_file, LA, RA, output_dir, temperature)
            param_file = os.path.join(os.path.dirname(__file__), 'parameters.cfg')
            process_intarna_queries(target_file, query_file, lunp_file, param_file, LA, RA, output_prefix, motif)
            post_process_features(target_file, output_dir)

            # Store mapping for CS_Dz (id2, seq2, target_file) in order
            for qname, qseq in queries:
                cs_dz_records.append({
                    'id2': qname,
                    'seq2': qseq,
                    'target_file': os.path.basename(target_file)
                })

        if processed_files:
            final_output_path = os.path.join(args.output_dir, "all_generated_merged_num.csv")
            # Also save a copy in the current working directory
            pwd_output_path = "all_generated_merged_num.csv"
            merge_all_generated_files(".", pwd_output_path, list(processed_files))
            # Patch the_feature_set_predicted.csv to use the correct id2, seq2, target_file columns from CS_Dz in order
            predicted_path = os.path.join(args.output_dir, "the_feature_set_predicted.csv")
            if os.path.exists(predicted_path):
                predicted_df = pd.read_csv(predicted_path)
                cs_dz_df = pd.DataFrame(cs_dz_records)
                # Overwrite the columns in the predicted output with the CS_Dz values in order
                for col in ['id2', 'seq2', 'target_file']:
                    if col in predicted_df.columns and col in cs_dz_df.columns:
                        predicted_df[col] = cs_dz_df[col].values
                predicted_df.to_csv(predicted_path, index=False)
                print(f"\n✓ Patched {predicted_path} with id2, seq2, target_file columns from CS_Dz in order.")
            print("\n✅ Feature generation completed successfully for target_screen mode!")
            return  # Skip the second processing phase

    elif args.feature_mode == 'target_check':
        for _, row in params_df.iterrows():
            LA, RA, CS, temperature, core = int(row['LA']), int(row['RA']), row['CS'].split(','), float(row['Tem']), row['CA']
            start_end_data = row['Start_End_Index'].split(':')
            if len(start_end_data) != 2:
                raise ValueError("Invalid format in Start_End_Index column. Expected format: 'target_file:start-end' (e.g., '1.fasta:50-100').")

            target_file_name, positions = start_end_data[0], start_end_data[1]
            try:
                start, end = map(int, positions.split('-'))
            except ValueError:
                raise ValueError(f"Invalid start-end positions in Start_End_Index column: {positions}. Expected format: 'start-end' (e.g., '50-100').")

            for fasta_file in args.targets.split(','):
                if os.path.basename(fasta_file) != target_file_name:
                    continue

                with open(fasta_file, 'r') as f:
                    target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

                if start < 0 or end > len(target_seq) or start >= end:
                    raise ValueError(f"Invalid start-end positions {start}-{end} for file {target_file_name}. Must be within the bounds of the target sequence and start < end.")

                print(f"Checking target region from position {start} to {end} in the target sequence of {target_file_name}.")
                motif_matches = find_CS(target_seq[start - 1:end], CS)

                # Adjust positions to be relative to the full sequence
                adjusted_matches = [
                    (start + match[0], start + match[1], match[2], start + match[3], start + match[4])
                    for match in motif_matches
                ]

                queries = prepare_sequences(target_seq, adjusted_matches, LA, RA, core)
                query_file = f"queries_{os.path.basename(target_file_name).split('.')[0]}_{start}_{end}.fasta"
                write_queries_to_fasta(queries, query_file)
                print(f"Generated {len(queries)} queries for motifs in region {start}-{end} of {target_file_name}.")

    elif args.feature_mode == 'specific_query':
        if not {'LA_seq', 'RA_seq', 'CS', 'CS_Index_query', 'Tem', 'CA'}.issubset(params_df.columns):
            raise ValueError("The CSV file must contain the columns: LA_seq, RA_seq, CS, CS_Index_query, Tem, and CA.")

        processed_files = set()
        cs_dz_records = []  # To store id2, seq2, target_file for each query in order
        for _, row in params_df.iterrows():
            LA_seq, RA_seq, CS, temperature, core = row['LA_seq'], row['RA_seq'], row['CS'], float(row['Tem']), row['CA']
            target_file, position = row['CS_Index_query'].split(':')
            processed_files.add(target_file)
            start, end = map(int, position.split('-'))

            with open(target_file, 'r') as f:
                target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

            # Validate the CS motif
            sequence_at_position = target_seq[start - 1:end]  # Adjust for 0-based indexing
            if sequence_at_position.upper() != CS.upper():
                print(f"Warning: CS motif '{CS}' does not match the sequence at position {start}-{end} in {target_file}. Found: '{sequence_at_position}'. Using the sequence from the target file.")
                CS = sequence_at_position.upper()

            # Generate query using LA_seq and RA_seq directly (no reverse complement)
            query_seq = f"{LA_seq}{core}{RA_seq}"
            queries = [(f"{CS}-{start}-{end}", query_seq)]
            query_file = f"queries_{os.path.basename(target_file).split('.')[0]}_{start}_{end}.fasta"
            write_queries_to_fasta(queries, query_file)
            print(f"Generated query file: {query_file}")

            output_prefix = os.path.basename(target_file).split('.')[0]
            output_dir = f"rnaplfold_output_{output_prefix}"
            os.makedirs(output_dir, exist_ok=True)
            lunp_file = run_rnaplfold(target_file, len(LA_seq), len(RA_seq), output_dir, temperature)
            param_file = os.path.join(os.path.dirname(__file__), 'parameters.cfg')
            process_intarna_queries(target_file, query_file, lunp_file, param_file, len(LA_seq), len(RA_seq), output_prefix, CS)
            post_process_features(target_file, output_dir)

            # Store mapping for CS_Dz (id2, seq2, target_file) in order
            for qname, qseq in queries:
                cs_dz_records.append({
                    'id2': qname,
                    'seq2': qseq,
                    'target_file': os.path.basename(target_file)
                })

        if processed_files:
            final_output_path = os.path.join(args.output_dir, "all_generated_merged_num.csv")
            # Also save a copy in the current working directory
            pwd_output_path = "all_generated_merged_num.csv"
            merge_all_generated_files(".", pwd_output_path, list(processed_files))
            # Patch the_feature_set_predicted.csv to use the correct id2, seq2, target_file columns from CS_Dz in order
            predicted_path = os.path.join(args.output_dir, "the_feature_set_predicted.csv")
            if os.path.exists(predicted_path):
                predicted_df = pd.read_csv(predicted_path)
                cs_dz_df = pd.DataFrame(cs_dz_records)
                # Overwrite the columns in the predicted output with the CS_Dz values in order
                for col in ['id2', 'seq2', 'target_file']:
                    if col in predicted_df.columns and col in cs_dz_df.columns:
                        predicted_df[col] = cs_dz_df[col].values
                predicted_df.to_csv(predicted_path, index=False)
                print(f"\n✓ Patched {predicted_path} with id2, seq2, target_file columns from CS_Dz in order.")
            print("\n✅ Feature generation completed successfully for specific_query mode!")
            return  # Skip the second processing phase

    else:
        raise ValueError("Invalid feature mode specified.")

    print("Query generation completed for all modes.")

    args.cfg = os.path.join(os.path.dirname(__file__), 'parameters.cfg')

    params_df = pd.read_csv(args.params)
    if not {'LA', 'RA', 'CS', 'Tem', 'CA'}.issubset(params_df.columns):
        print("The CSV file must contain the columns: LA, RA, CS, Tem, and CA.")
        return

    if os.path.isdir(args.targets):
        fasta_files = [os.path.join(args.targets, f) for f in os.listdir(args.targets) if f.endswith('.fasta')]
    else:
        fasta_files = args.targets.split(',')

    if not fasta_files:
        print("No FASTA files found. Please provide valid input.")
        return

    for _, row in params_df.iterrows():
        LA = int(row['LA'])
        RA = int(row['RA'])
        CS = row['CS'].split(',')
        temperature = float(row['Tem'])
        core = row['CA']

        for fasta_file in fasta_files:
            print(f"\nProcessing file: {fasta_file}")
            with open(fasta_file, 'r') as f:
                target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

            output_dir = f"rnaplfold_output_{os.path.basename(fasta_file).split('.')[0]}"
            lunp_file = run_rnaplfold(fasta_file, LA, RA, output_dir, temperature)

            if args.feature_mode == 'default':
                motif_matches = find_CS(target_seq, CS)
                queries = prepare_sequences(target_seq, motif_matches, LA, RA, core)
            elif args.feature_mode == 'target_screen':
                if 'CS_index' not in row:
                    raise ValueError("For target_screen mode, the CSV file must contain a 'CS_index' column with the format 'target_file:CS_index' (e.g., '1.fasta:17').")
                
                cs_index_data = row['CS_index'].split(':')
                if len(cs_index_data) != 2:
                    raise ValueError("Invalid format in CS_index column. Expected format: 'target_file:CS_index' (e.g., '1.fasta:17').")
                
                target_file_name, region = cs_index_data[0], cs_index_data[1]
                try:
                    start, end = map(int, region.split('-'))
                except ValueError:
                    raise ValueError(f"Invalid CS_index format: {region}. Expected format: 'start-end' (e.g., '17-18').")

                if os.path.basename(fasta_file) != target_file_name:
                    continue

                if start < 0 or end > len(target_seq) or start >= end:
                    raise ValueError(f"Invalid CS_index range {start}-{end} for file {target_file_name}. Must be within the bounds of the target sequence and start < end.")
                
                motif = CS[0]
                sequence_at_position = target_seq[start: end]
                if sequence_at_position.upper() != motif.upper():
                    print(f"Warning: CS motif '{motif}' does not match the sequence at position {start} in {target_file_name}. Found: '{sequence_at_position}'. Using the sequence from the target file.")
                    motif = sequence_at_position.upper()

                id2 = f"{start}-{end}"
                if int(id2.split('-')[0]) != start:
                    raise ValueError(f"CS_index {start} does not match the first number of id2 {id2.split('-')[0]}.")

                print(f"Validated CS motif and corrected id2 calculation: {id2}")

                motif_matches = [(start, end, motif, start, end)]
                queries = prepare_sequences(target_seq, motif_matches, LA, RA, core)
            elif args.feature_mode == 'target_check':
                if 'Start_End_Index' not in row:
                    raise ValueError("For target_check mode, the CSV file must contain a 'Start_End_Index' column with the format 'target_file:start-end' (e.g., '1.fasta:50-100').")
                
                start_end_data = row['Start_End_Index'].split(':')
                if len(start_end_data) != 2:
                    raise ValueError("Invalid format in Start_End_Index column. Expected format: 'target_file:start-end' (e.g., '1.fasta:50-100').")
                
                target_file_name, positions = start_end_data[0], start_end_data[1]
                if os.path.basename(fasta_file) != target_file_name:
                    continue
                
                try:
                    start, end = map(int, positions.split('-'))
                except ValueError:
                    raise ValueError(f"Invalid start-end positions in Start_End_Index column: {positions}. Expected format: 'start-end' (e.g., '50-100').")
                
                if start < 0 or end >= len(target_seq) or start >= end:
                    raise ValueError(f"Invalid start-end positions {start}-{end} for file {target_file_name}. Must be within the bounds of the target sequence and start < end.")
                
                print(f"Checking target region from position {start} to {end} in the target sequence of {target_file_name}.")
                motif_matches = find_CS(target_seq[start - 1:end], CS)

                # Adjust positions to be relative to the full sequence
                adjusted_matches = [
                    (start + match[0], start + match[1], match[2], start + match[3], start + match[4])
                    for match in motif_matches
                ]

                queries = prepare_sequences(target_seq, adjusted_matches, LA, RA, core)
                query_file = f"queries_{os.path.basename(target_file_name).split('.')[0]}_{start}_{end}.fasta"
                write_queries_to_fasta(queries, query_file)
                print(f"Generated {len(queries)} queries for motifs in region {start}-{end} of {target_file_name}.")

            elif args.feature_mode == 'target_screen':
                # Skip the second processing phase since it's already handled
                continue

            elif args.feature_mode == 'specific_query':
                # Skip the second processing phase since it's already handled
                continue

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            query_file = f"queries_{os.path.basename(fasta_file).split('.')[0]}_{timestamp}.fasta"
            write_queries_to_fasta(queries, query_file)
            print(f"Generated {len(queries)} queries to {query_file}")

            output_prefix = os.path.basename(fasta_file).split('.')[0]
            process_intarna_queries(fasta_file, query_file, lunp_file, args.cfg, LA, RA, output_prefix, CS[0])
            
            post_process_features(fasta_file, output_dir)

    for file in os.listdir():
        if file.startswith("queries_") and file.endswith(".fasta"):
            os.remove(file)
            print(f"✓ Deleted {file}")

    print("\n✅ Feature generation completed successfully for all files!")

    # Always write the merged file to the output directory
    final_output_path = os.path.join(args.output_dir, "all_generated_merged_num.csv")
    merge_all_generated_files(output_dir=args.output_dir, final_output_file=final_output_path, targets_fasta_files=fasta_files)

if __name__ == "__main__":
    # Always use the main() parser for CLI entry
    main()