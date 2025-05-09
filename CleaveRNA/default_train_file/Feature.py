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

def convert_t_to_u(sequence):
    return sequence.replace('T', 'U')

def run_rnaplfold(target_file, LA, RA, output_dir, temperature):
    os.makedirs(output_dir, exist_ok=True)  # Create a unique directory for each file
    u_length = LA + RA + 4
    target_file_path = os.path.abspath(target_file)
    command = f"RNAplfold -W 150 -L 100 -u {u_length} -T {temperature} < {target_file_path}"
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

def process_specific_target(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    if not {'name', 'sequence'}.issubset(df.columns):
        raise ValueError("The CSV file for specific_target mode must contain 'name' and 'sequence' columns.")
    
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for _, row in df.iterrows():
        query_name = row['name']
        query_sequence = row['sequence']
        query_file = os.path.join(output_dir, f"{query_name}.fasta")
        with open(query_file, "w") as f:
            f.write(f">{query_name}\n{query_sequence}\n")
        print(f"Query FASTA file created: {query_file}")

def construct_intarna_command(query_file, target_file, param_file, additional):
    return (
        f"IntaRNA --query {query_file} --target {target_file} "
        f"--qIntLenMax 0 --parameterFile {param_file} "
        f"--outMode C --outNumber 1 --outOverlap N "
        f"--outCsvCols 'id2,seq2,E,ED1,ED2,Pu1,Pu2,subseqDB,hybridDB,E_hybrid,"
        f"seedStart1,seedEnd1,seedStart2,seedEnd2,seedE,seedED1,seedED2,seedPu1,seedPu2,P_E' "
        + additional
    )

def process_intarna_queries(target_file, query_file, lunp_file, param_file, left_arm, right_arm, output_prefix):
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
                subprocess.run(cmd, shell=True, check=True)
            all_results.extend([f"{output_prefix}_result_{i}_with_region.csv", f"{output_prefix}_result_{i}_without_region.csv", f"{output_prefix}_result_{i}_pairwise.csv"])
        os.remove(tmp_q)

    final_files = [f"{output_prefix}_Results_with_region.csv", f"{output_prefix}_Results_without_region.csv", f"{output_prefix}_Results_pairwise.csv"]
    for out, res_group in zip(final_files, [all_results[i::3] for i in range(3)]):
        with open(out, "w") as outf:
            if res_group:
                with open(res_group[0]) as first: outf.write(first.readline())
                for r in res_group:
                    with open(r) as f: next(f); outf.write(f.read())
    for f in os.listdir():
        if f.startswith(f"{output_prefix}_result_") and f.endswith(".csv"):
            os.remove(f)
    
    # After creating the final files, merge numerical columns
    merge_numerical_columns(output_prefix=output_prefix)

def merge_numerical_columns(output_file=None, output_prefix=None):
    """
    Merge numerical columns from multiple result files
    """
    if output_prefix is None:
        print("No output prefix provided for merge_numerical_columns, using default")
        output_prefix = "5"  # This was the hardcoded default

    print(f"\nMerging numerical columns with prefix {output_prefix}...")

    # Define the specific files to merge based on the output_prefix
    result_files = [
        f"{output_prefix}_Results_with_region.csv",
        f"{output_prefix}_Results_without_region.csv",
        f"{output_prefix}_Results_pairwise.csv"
    ]

    merged_data = pd.DataFrame()

    for idx, file in enumerate(result_files, start=1):
        if os.path.exists(file):
            print(f"Processing file: {file}")  # Debug statement
            try:
                data = pd.read_csv(file)
                print(f"File content preview:\n{data.head()}\n")  # Debug statement

                if data.empty:
                    print(f"Warning: File {file} is empty")
                    continue

                numerical_cols = data.select_dtypes(include=['number'])
                numerical_cols = numerical_cols.add_suffix(f"_{idx}")

                if 'id2' in data.columns:
                    numerical_cols.insert(0, 'id2', data['id2'])

                merged_data = pd.concat([merged_data, numerical_cols], axis=1)
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
        else:
            print(f"File not found: {file}")

    if not merged_data.empty:
        print(f"✓ Merged numerical columns processed successfully.")
        print(f"Merged data preview:\n{merged_data.head()}\n")
    else:
        print("✗ No numerical columns found to merge.")

    # Create numerical_columns DataFrame properly
    numerical_columns = merged_data.select_dtypes(include=['number'])
    
    # Ensure 'id2' is a single column before inserting
    if 'id2' in merged_data.columns and isinstance(merged_data['id2'], pd.Series):
        numerical_columns.insert(0, 'id2', merged_data['id2'])
    else:
        print("Warning: 'id2' is not a single column or is missing.")



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
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"✗ Result file {result_file} not found. Did IntaRNA run correctly?")

            print(f"✓ Processing {result_file}...")
            df = pd.read_csv(result_file)
            print(f"Result file {i} sample:\n", df.head())

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

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)

        # Save full merged data with file identifier
        full_output_path = os.path.join(output_dir, f"{base_filename}_generated_merged.csv")
        merged_data.to_csv(full_output_path, index=False)
        print(f"✓ Saved full merged data to {full_output_path}")
        print("Full data sample:\n", merged_data.head())

        # Save id2 and all numerical columns to generated_merged_num.csv
        numeric_columns = merged_data.select_dtypes(include=['number'])

        # Ensure id2 column is included 
        if 'id2' in merged_data.columns:
            # Create a new DataFrame with id2 first, then all numeric columns
            numeric_with_id = pd.DataFrame()
            numeric_with_id['id2'] = merged_data['id2']

            # Add all numeric columns
            for col in numeric_columns.columns:
                numeric_with_id[col] = numeric_columns[col]

            # Save to the output file
            num_output_path = os.path.join(output_dir, "generated_merged_num.csv")
            numeric_with_id.to_csv(num_output_path, index=False)
            print(f"✓ Saved id2 and numerical columns to {num_output_path}")
            print("Numerical data sample:\n", numeric_with_id.head())
        else:
            print("✗ Warning: 'id2' column not found in merged data")

        # Move the generated_merged_num.csv from the rnaplfold directory to the working directory
        source_file = os.path.join(output_dir, "generated_merged_num.csv")
        destination_file = os.path.join(os.getcwd(), "generated_merged_num.csv")
        if os.path.exists(source_file):
            os.rename(source_file, destination_file)
            print(f"✓ Moved {source_file} to {destination_file}")
        else:
            print(f"✗ File {source_file} does not exist in the rnaplfold directory.")

    except Exception as e:
        print(f"✗ Post-processing failed: {e}")

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--targets', required=True, help="Path to a directory or a comma-separated list of FASTA files")
        parser.add_argument('--params', required=True, help="Path to the CSV file containing LA, RA, CS, temperature, and core")
        parser.add_argument('--mode_feature', required=True, choices=['default', 'target_screen', 'target_check', 'specific_target'], help="Mode of operation")
        parser.add_argument('--specific_csv', help="CSV file for specific_target mode")
        args = parser.parse_args()

    args.cfg = os.path.join(os.path.dirname(__file__), 'parameters.cfg')

    params_df = pd.read_csv(args.params)
    if not {'LA', 'RA', 'CS', 'temperature', 'core'}.issubset(params_df.columns):
        print("The CSV file must contain the columns: LA, RA, CS, temperature, and core.")
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
        temperature = float(row['temperature'])
        core = row['core']

        for fasta_file in fasta_files:
            print(f"\nProcessing file: {fasta_file}")
            with open(fasta_file, 'r') as f:
                target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

            output_dir = f"rnaplfold_output_{os.path.basename(fasta_file).split('.')[0]}"
            lunp_file = run_rnaplfold(fasta_file, LA, RA, output_dir, temperature)
            unpaired_probs = parse_rnaplfold_output(lunp_file)

            if args.mode_feature == 'default':
                motif_matches = find_CS(target_seq, CS)
                queries = prepare_sequences(target_seq, motif_matches, LA, RA, core)
            elif args.mode_feature == 'target_screen':
                # Parse the CS_index column to get the target file name and the CS index
                if 'CS_index' not in row:
                    raise ValueError("For target_screen mode, the CSV file must contain a 'CS_index' column with the format 'target_file:CS_index' (e.g., '1.fasta:17').")
                
                # Extract the target file name and CS index from the CS_index column
                cs_index_data = row['CS_index'].split(':')
                if len(cs_index_data) != 2:
                    raise ValueError("Invalid format in CS_index column. Expected format: 'target_file:CS_index' (e.g., '1.fasta:17').")
                
                target_file_name, position = cs_index_data[0], int(cs_index_data[1])
                if os.path.basename(fasta_file) != target_file_name:
                    continue  # Skip this row if the target file does not match the current FASTA file
                
                if position < 0 or position >= len(target_seq):
                    raise ValueError(f"Invalid CS position {position} for file {target_file_name}. Must be between 0 and {len(target_seq) - 1}.")
                
                motif = CS[0]  # Assuming only one motif is provided in the CS column
                print(f"Screening CS motif '{motif}' at position {position} in the target sequence of {target_file_name}.")
                motif_matches = [(position, position + len(motif), motif, position, position + len(motif))]
                queries = prepare_sequences(target_seq, motif_matches, LA, RA, core)
            elif args.mode_feature == 'target_check':
                # Parse the Start_End_target column to get the target file name and the start-end positions
                if 'Start_End_target' not in row:
                    raise ValueError("For target_check mode, the CSV file must contain a 'Start_End_target' column with the format 'target_file:start-end' (e.g., '1.fasta:50-100').")
                
                # Extract the target file name and start-end positions from the Start_End_target column
                start_end_data = row['Start_End_target'].split(':')
                if len(start_end_data) != 2:
                    raise ValueError("Invalid format in Start_End_target column. Expected format: 'target_file:start-end' (e.g., '1.fasta:50-100').")
                
                target_file_name, positions = start_end_data[0], positions[1]
                if os.path.basename(fasta_file) != target_file_name:
                    continue  # Skip this row if the target file does not match the current FASTA file
                
                # Extract the start and end positions
                try:
                    start, end = map(int, positions.split('-'))
                except ValueError:
                    raise ValueError(f"Invalid start-end positions in Start_End_target column: {positions}. Expected format: 'start-end' (e.g., '50-100').")
                
                if start < 0 or end >= len(target_seq) or start >= end:
                    raise ValueError(f"Invalid start-end positions {start}-{end} for file {target_file_name}. Must be within the bounds of the target sequence and start < end.")
                
                print(f"Checking target region from position {start} to {end} in the target sequence of {target_file_name}.")
                motif_matches = [(start, end, "region", start, end)]
                queries = prepare_sequences(target_seq, motif_matches, LA, RA, core)
            elif args.mode_feature == 'specific_target':
                if not args.specific_csv:
                    raise ValueError("You must provide a CSV file with --specific_csv for specific_target mode.")
                process_specific_target(args.specific_csv, output_dir)
                continue

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            query_file = f"queries_{os.path.basename(fasta_file).split('.')[0]}_{timestamp}.fasta"
            write_queries_to_fasta(queries, query_file)
            print(f"Generated {len(queries)} queries to {query_file}")

            output_prefix = os.path.basename(fasta_file).split('.')[0]
            process_intarna_queries(fasta_file, query_file, lunp_file, args.cfg, LA, RA, output_prefix)
            
            # Call post_process_features for each file
            post_process_features(fasta_file, output_dir)

    # Delete all generated query FASTA files at the end
    for file in os.listdir():
        if file.startswith("queries_") and file.endswith(".fasta"):
            os.remove(file)
            print(f"✓ Deleted {file}")

    print("\n✅ Feature generation completed successfully for all files!")

if __name__ == "__main__":
    main()