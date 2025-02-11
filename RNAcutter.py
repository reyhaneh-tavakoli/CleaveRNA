import sys
import subprocess
import re
import time
import os
import pandas as pd
import pathlib

# Function to convert T to U in a sequence
def convert_t_to_u(sequence):
    return sequence.replace('T', 'U')

# Step 1: Calculate unpaired probabilities using RNAplfold
def run_rnaplfold(target_file):
    left_arm_length = int(sys.argv[2].split("=")[1])
    right_arm_length = int(sys.argv[3].split("=")[1])
    u_length = left_arm_length + right_arm_length + 4  # no catalytic core in target
    command = f"RNAplfold -W 150 -L 100 -u {u_length} < {target_file}"
    subprocess.run(command, shell=True)

# Step 2: Parse RNAplfold output and find GC and AC motifs
def parse_rnaplfold_output(output_file):
    with open(output_file, "r") as file:
        lines = file.readlines()

    unpaired_probs = {}
    for line in lines:
        if line.startswith("#") or line.startswith("i$"):
            continue
        parts = line.strip().split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            position = int(parts[0])
            prob = float(parts[1])
            unpaired_probs[position] = prob
        except ValueError as e:
            print(f"Skipping line due to error: {line} - {e}")
    return unpaired_probs

def find_motifs(sequence):
    matches = []
    used_positions = set()
    
    # Check if motifs are provided
    if len(sys.argv) < 2 or not sys.argv[1].startswith("-motifs"):
        print("Error: Please provide motifs using the -motifs command.")
        sys.exit(1)

    # Extract motifs from command line argument
    motifs_arg = sys.argv[1].split("=")[1]
    motifs_list = motifs_arg.split(",")

    # Find all specified motifs and track their exact positions
    for motif in motifs_list:
        for match in re.finditer(motif, sequence):
            start = match.start()
            pos1 = start + 1  # First nucleotide position
            pos2 = start + 2  # Second nucleotide position
            position_key = f"{pos1}-{pos2}"
            
            # Only use positions that haven't been claimed yet
            if position_key not in used_positions:
                used_positions.add(position_key)
                start_region = max(0, start - 10)
                end_region = min(len(sequence), match.end() + 10)
                matches.append((start_region, end_region, match.group(), pos1, pos2))
    
    return matches

# Define valid nucleotides
valid_nucleotides = {"A", "U", "C", "G"}

def prepare_sequences(sequence, motifs):
    linker = "ggcuagcuacaacga" # Linker sequence (valid RNA sequence)
    queries = []
    filtered_motifs = []
    
    for start, end, motif, pos1, pos2 in motifs:
        if pos1 > 8 and pos2 > 8:
            filtered_motifs.append((start, end, motif, pos1, pos2))
    
    for start, end, motif, pos1, pos2 in filtered_motifs:
        # Extract lengths from command line arguments
        left_arm_length = int(sys.argv[2].split("=")[1])
        right_arm_length = int(sys.argv[3].split("=")[1])

        #Consider the second nucleotide of the cleavage site and right_arm_length nucleotides after
        left_arm = sequence[pos1 : pos1 + left_arm_length ]

        # Consider left_arm_length nucleotides before the first nucleotide of the cleavage site
        right_arm = sequence[ pos2 - (right_arm_length + 3)  : pos2 - 2 ]

        # Check if left arm and right arm lengths are provided
        if len(sys.argv) < 4 or not sys.argv[2].startswith("-LA") or not sys.argv[3].startswith("-RA"):
            print("Error: Please provide lengths for left arm (-LA) and right arm (-RA).")
            sys.exit(1)

        if not set(left_arm).issubset(valid_nucleotides) or not set(right_arm).issubset(valid_nucleotides):
            continue
            
        # Generate the reverse complement sequences
        complementary_left_arm = "".join(["AUCG"["UAGC".index(n)] for n in left_arm][::-1])
        complementary_right_arm = "".join(["AUCG"["UAGC".index(n)] for n in right_arm][::-1])
        
        # Construct the query sequence
        query_sequence = f"{complementary_left_arm}{linker}{complementary_right_arm}"
        query_name = f"{motif}-{pos1}-{pos2}"  # Using actual dinucleotide positions
        queries.append((query_name, query_sequence))
    return queries

def write_queries_to_fasta(queries, query_file):
    with open(query_file, "w") as f:
        for i, (query_name, query_sequence) in enumerate(queries):
            f.write(f">{query_name}\n{query_sequence}\n")

def construct_intarna_command(query_file, target_file, parameter_file, additional_params):
    base_command = (
        f"IntaRNA "
        f"--query {query_file} "
        f"--target {target_file} "
        f"--qIntLenMax 0 "
        f"--parameterFile {parameter_file} "
        f"--outMode C "
        f"--outMaxE -4 "
        f"--outNumber 2 "
        f"--outOverlap N "
        f"--outCsvCols 'id2,seq2,E,Etotal,ED1,ED2,Pu1,Pu2,subseqDB,hybridDB,E_hybrid,seedStart1,seedEnd1,seedStart2,seedEnd2,seedE,seedED1,seedED2,seedPu1,seedPu2,P_E' "
    )
    return base_command + additional_params

def process_intarna_queries(target_file, query_file, unpaired_prob_file, parameter_file):
    # Get sequence length
    with open(target_file, "r") as f:
        sequence = "".join(line.strip() for line in f if not line.startswith(">"))
    seq_length = len(sequence)

    # Read queries from fasta file
    queries = []
    with open(query_file, "r") as f:
        current_name = ""
        current_seq = ""
        for line in f:
            if line.startswith(">"):
                if current_name and current_seq:
                    queries.append((current_name, current_seq))
                current_name = line.strip()[1:]
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_name and current_seq:
            queries.append((current_name, current_seq))

    all_results = []
    for i, (query_name, query_seq) in enumerate(queries, 1):
        # Write each query sequence to a temporary file
        temp_query_file = f"temp_query_{i}.fasta"
        with open(temp_query_file, "w") as temp_file:
            temp_file.write(f">{query_name}\n{query_seq}\n")

        # Extract motif positions from query name and adjust for 1-based indexing
        motif_info = query_name.split("-")
        if len(motif_info) >= 3:
            start_pos = max(1, int(motif_info[1]) - 9)  # Ensure minimum is 1
            end_pos = min(seq_length, int(motif_info[2]) + 11)  # Ensure maximum is sequence length
            target_region = f"{start_pos}-{end_pos}"

            # First IntaRNA call with tRegion
            left_arm_length = int(sys.argv[2].split("=")[1])
            right_arm_length = int(sys.argv[3].split("=")[1])
            total_length = left_arm_length + right_arm_length + 3
            additional_params1 = (
                f"--tAcc P "
                f"--tIntLenMax {total_length} "
                f"--tAccFile {unpaired_prob_file} "
                f"--tRegion {target_region} "
                f"--out result_{i}_with_region.csv "
            )
            command1 = construct_intarna_command(temp_query_file, target_file, parameter_file, additional_params1)

            # Second IntaRNA call without tRegion
            additional_params2 = (
                f"--tAcc P "
                f"--tIntLenMax {total_length} "
                f"--tAccFile {unpaired_prob_file} "
                f"--out result_{i}_without_region.csv "
            )
            command2 = construct_intarna_command(temp_query_file, target_file, parameter_file, additional_params2)

            # Third IntaRNA call with query as both target and query
            # ensure there is a prediction for each pair (outMaxE), to avoid missing predictions
            additional_params3 = (
                f"--out result_{i}_pairwise.csv "
                f"--outMaxE 10 "
            )
            command3 = construct_intarna_command(temp_query_file, temp_query_file, parameter_file, additional_params3)

            print(f"Processing query {i}/{len(queries)}: {query_name}")
            subprocess.run(command1, shell=True, check=True)
            subprocess.run(command2, shell=True, check=True)
            subprocess.run(command3, shell=True, check=True)

            all_results.append(f"result_{i}_with_region.csv")
            all_results.append(f"result_{i}_without_region.csv")
            all_results.append(f"result_{i}_pairwise.csv")

        # Remove the temporary query file
        os.remove(temp_query_file)

    # Modify the output file names in main()
    output_file_call1 = "Results_with_region.csv"
    output_file_call2 = "Results_without_region.csv"
    output_file_call3 = "Results_pairwise.csv"

    # Process results separately for each call
    with (
        open(output_file_call1, "w") as outfile1,
        open(output_file_call2, "w") as outfile2,
        open(output_file_call3, "w") as outfile3,
    ):
        # Write header for first call results
        with open(all_results[0], "r") as firstfile:
            header = firstfile.readline()
            outfile1.write(header)
            outfile2.write(header)
            outfile3.write(header)

        # Write results from first calls to first file
        for i in range(0, len(all_results), 3):
            with open(all_results[i], "r") as infile:
                next(infile)  # skip header
                outfile1.write(infile.read())

        # Write results from second calls to second file
        for i in range(1, len(all_results), 3):
            with open(all_results[i], "r") as infile:
                next(infile)  # skip header
                outfile2.write(infile.read())
                
        # Write results from third calls to third file
        for i in range(2, len(all_results), 3):
            with open(all_results[i], "r") as infile:
                next(infile)  # skip header
                outfile3.write(infile.read())

    # Remove other CSV files but keep the specified result files
    for file in os.listdir("."):
        if file.endswith(".csv") and file not in ["Results_with_region.csv", "Results_without_region.csv", "Results_pairwise.csv"]:
            os.remove(file)

def cleanup_temp_files():
    temp_files = ["temp_sequence.fasta", "output_queries.fasta"]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)

def main():
    if len(sys.argv) < 5 or not sys.argv[4].startswith("-target"):
        print("Error: Please provide the target file using the -target command.")
        sys.exit(1)
    target_file = sys.argv[4].split("=")[1]
    query_file = "queries.fasta"
    unpaired_prob_file = "converted_sequence_lunp"
    
    if len(sys.argv) < 6:
        print("Error: Please provide the parameter file using the -cfg command.")
        sys.exit(1)
    if not sys.argv[5].startswith("-cfg"):
        print("Error: Invalid parameter file argument. Please use the -cfg command.")
        sys.exit(1)
    parameter_file = sys.argv[5].split("=")[1]

    start_time = time.time()
    print(f"Start time: {time.ctime(start_time)}")

    # Read and convert sequence
    with open(target_file, "r") as file:
        lines = file.readlines()
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
        sequence = convert_t_to_u(sequence)

    # Write converted sequence to a temporary file
    temp_target_file = "temp_sequence.fasta"
    with open(temp_target_file, "w") as file:
        file.write(">converted_sequence\n")
        file.write(sequence + "\n")

    run_rnaplfold(temp_target_file)

    with open(target_file, "r") as file:
        lines = file.readlines()
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
        sequence = convert_t_to_u(sequence)

    motifs = find_motifs(sequence)
    queries = prepare_sequences(sequence, motifs)

    query_file = "output_queries.fasta"
    write_queries_to_fasta(queries, query_file)

    process_intarna_queries(target_file, query_file, unpaired_prob_file, parameter_file)

    cleanup_temp_files()

    end_time = time.time()
    print(f"End time: {time.ctime(end_time)}")
    print(f"Total run time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
