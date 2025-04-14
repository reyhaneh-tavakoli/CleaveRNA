#!/usr/bin/env python3

import sys
import subprocess
import re
import time
import os
import pandas as pd
import argparse
from datetime import datetime

def convert_t_to_u(sequence):
    return sequence.replace('T', 'U')

def run_rnaplfold(target_file, left_arm_length, right_arm_length, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create a unique directory for each file
    u_length = left_arm_length + right_arm_length + 4
    command = f"RNAplfold -W 150 -L 100 -u {u_length} < {target_file}"
    print(f"\nExecuting RNAplfold command: {command} in {output_dir}")
    
    try:
        # Run RNAplfold in the specified output directory
        subprocess.run(command, shell=True, check=True, cwd=output_dir)
        lunp_files = [f for f in os.listdir(output_dir) if f.endswith('_lunp')]
        if not lunp_files:
            raise FileNotFoundError("RNAplfold did not generate a _lunp file.")
        return os.path.join(output_dir, lunp_files[0])  # Return the full path to the _lunp file
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

def find_motifs(sequence, motifs_list):
    matches, used_positions = [], set()
    for motif in motifs_list:
        for match in re.finditer(motif, sequence):
            start = match.start()
            pos1, pos2 = start + 1, start + 2
            key = f"{pos1}-{pos2}"
            if key not in used_positions:
                used_positions.add(key)
                matches.append((max(0, start - 10), min(len(sequence), match.end() + 10), match.group(), pos1, pos2))
    return matches

def prepare_sequences(sequence, motifs, left_arm_length, right_arm_length):
    linker = "ggcuagcuacaacga"
    valid = {"A", "U", "C", "G"}
    queries = []
    for start, end, motif, pos1, pos2 in motifs:
        if pos1 <= 8 or pos2 <= 8:
            continue
        left_arm = sequence[pos1:pos1 + left_arm_length]
        right_arm = sequence[pos2 - right_arm_length - 3:pos2 - 2]
        if not set(left_arm).issubset(valid) or not set(right_arm).issubset(valid):
            continue
        comp = lambda s: ''.join(["AUCG"["UAGC".index(n)] for n in s][::-1])
        query_seq = f"{comp(left_arm)}{linker}{comp(right_arm)}"
        queries.append((f"{motif}-{pos1}-{pos2}", query_seq))
    return queries

def write_queries_to_fasta(queries, query_file):
    with open(query_file, "w") as f:
        for name, seq in queries:
            f.write(f">{name}\n{seq}\n")

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

def generate_feature_files(output_prefix):
    os.makedirs("feature_outputs", exist_ok=True)
    dfs = []
    for f in [f"{output_prefix}_Results_with_region.csv", f"{output_prefix}_Results_without_region.csv", f"{output_prefix}_Results_pairwise.csv"]:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df.add_suffix(f"_{len(dfs)+1}").rename(columns={f'seq2_{len(dfs)+1}': 'seq2'}))
        else:
            raise FileNotFoundError(f"Missing {f}")
    merged = pd.merge(pd.merge(dfs[0], dfs[1], on='seq2', how='outer'), dfs[2], on='seq2', how='outer')
    merged.to_csv(f"feature_outputs/{output_prefix}_merged_features.csv", index=False)

    feature_sets = {
        'feature_set_1': ['E_1', 'Pu1_1', 'E_hybrid_1', 'seedNumber_1', 'seedEbest_1', 'seedNumber_3'],
        'feature_set_2': ['E_1', 'Pu1_1', 'Pu2_1', 'E_hybrid_1', 'seedEbest_1', 'seedNumber_3']
    }
    for name, feats in feature_sets.items():
        cols = [f for f in feats if f in merged.columns]
        if cols:
            merged[cols].to_csv(f"feature_outputs/{output_prefix}_{name}.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--targets', required=True, help="Path to a directory or a comma-separated list of FASTA files")
    parser.add_argument('--motifs', required=True)
    parser.add_argument('--LA', type=int, required=True)
    parser.add_argument('--RA', type=int, required=True)
    args = parser.parse_args()

    # Determine if the input is a directory or a list of files
    if os.path.isdir(args.targets):
        fasta_files = [os.path.join(args.targets, f) for f in os.listdir(args.targets) if f.endswith('.fasta')]
    else:
        fasta_files = args.targets.split(',')

    if not fasta_files:
        print("No FASTA files found. Please provide valid input.")
        return

    for fasta_file in fasta_files:
        print(f"\nProcessing file: {fasta_file}")
        with open(fasta_file, 'r') as f:
            target_seq = convert_t_to_u(''.join([line.strip() for line in f if not line.startswith(">")]))

        # Create a unique output directory for RNAplfold
        output_dir = f"rnaplfold_output_{os.path.basename(fasta_file).split('.')[0]}"
        lunp_file = run_rnaplfold(fasta_file, args.LA, args.RA, output_dir)
        unpaired_probs = parse_rnaplfold_output(lunp_file)

        motif_matches = find_motifs(target_seq, args.motifs.split(','))
        print(f"Found {len(motif_matches)} motif occurrences in {fasta_file}")

        queries = prepare_sequences(target_seq, motif_matches, args.LA, args.RA)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        query_file = f"queries_{os.path.basename(fasta_file).split('.')[0]}_{timestamp}.fasta"
        write_queries_to_fasta(queries, query_file)
        print(f"Generated {len(queries)} queries to {query_file}")

        output_prefix = os.path.basename(fasta_file).split('.')[0]
        process_intarna_queries(fasta_file, query_file, lunp_file, args.cfg, args.LA, args.RA, output_prefix)

        generate_feature_files(output_prefix)

    print("\n✅ Feature generation completed successfully for all files!")

if __name__ == "__main__":
    main()
