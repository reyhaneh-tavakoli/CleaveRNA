import sys
import subprocess
import re
import time
import os


# Step 1: Calculate unpaired probabilities using RNAplfold
def run_rnaplfold(sequence_file):
    command = f"RNAplfold -W 150 -L 100 -u 35 < {sequence_file}"
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
    
    # Find all AC/GC motifs and track their exact positions
    for match in re.finditer(r"AC|GC|GU|AU", sequence):
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

# Step 4: Prepare sequences for each cleavage site
def prepare_sequences(sequence, motifs):
    linker = "GGCUAGCUACAACGA"
    queries = []
    
    for start, end, motif, pos1, pos2 in motifs:
        # Consider 9 nucleotides before the first nucleotide of the cleavage site
        left_arm = sequence[pos1 : pos1 + 9]
        # Consider the second nucleotide of the cleavage site and 8 nucleotides after
        right_arm = sequence[start + 1 : start + 10]
        
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


def process_intarna_queries(
    target_file, query_file, unpaired_prob_file, parameter_file
):
    # Get sequence length
    with open(target_file, "r") as f:
        sequence = "".join(line.strip() for line in f if not line.startswith(">"))
    seq_length = len(sequence)

    # Create temporary directory for individual query files
    temp_dir = "temp_queries"
    os.makedirs(temp_dir, exist_ok=True)

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

    def construct_intarna_command(target_file, unpaired_prob_file, temp_query_file, parameter_file, temp_dir, i, additional_params):
        base_command = (
            f"IntaRNA "
            f"--target {target_file} "
            f"--tAcc P "
            f"--tAccFile {unpaired_prob_file} "
            f"--tIntLenMax 34 "
            f"--query {temp_query_file} "
            f"--qIntLenMax 0 "
            f"--parameterFile {parameter_file} "
            f"--outMode C "
            f"--outMaxE -4 "
            f"--outNumber 2 "
            f"--outCsvCols 'id2,seq2,E,Etotal,ED1,ED2,Pu1,Pu2,subseqDB,hybridDB,Pu2,E_dangleL,E_dangleR,E_endL,E_endR,E_init,E_loops,E_hybrid,E_norm,E_hybridNorm,E_add,seedStart1,seedEnd1,seedStart2,seedEnd2,seedE,seedED1,seedED2,seedPu1,Eall2,Zall,Zall1,Zall2,EallTotal,seedPu2,w,Eall,Eall1,P_E,RT' "
        )
        return base_command + additional_params

    all_results = []
    for i, (query_name, query_seq) in enumerate(queries, 1):
        # Write individual query to temporary file
        temp_query_file = f"{temp_dir}/query_{i}.fasta"
        with open(temp_query_file, "w") as f:
            f.write(f">{query_name}\n{query_seq}\n")

        # Extract motif positions from query name and adjust for 1-based indexing
        motif_info = query_name.split("-")
        if len(motif_info) >= 3:
            start_pos = max(1, int(motif_info[1]) - 9)  # Ensure minimum is 1
            end_pos = min(seq_length, int(motif_info[2]) + 11)  # Ensure maximum is sequence length
            target_region = f"{start_pos}-{end_pos}"

            # First IntaRNA call with tRegion
            additional_params1 = (
                f"--tRegion {target_region} "
                f"--out {temp_dir}/result_{i}_with_region.csv "
            )
            command1 = construct_intarna_command(target_file, unpaired_prob_file, temp_query_file, parameter_file, temp_dir, i, additional_params1)

            # Second IntaRNA call without tRegion
            additional_params2 = (
                f"--out {temp_dir}/result_{i}_without_region.csv "
            )
            command2 = construct_intarna_command(target_file, unpaired_prob_file, temp_query_file, parameter_file, temp_dir, i, additional_params2)

            # Third IntaRNA call with --outPairwise pairwise_matrix.csv
            additional_params3 = (
                f"--outPairwise 1 "
                f"--out {temp_dir}/result_{i}_pairwise.csv "   
            )
            command3 = construct_intarna_command(temp_query_file, unpaired_prob_file, temp_query_file, parameter_file, temp_dir, i, additional_params3)

            print(f"Processing query {i}/{len(queries)}: {query_name}")
            subprocess.run(command1, shell=True, check=True)
            subprocess.run(command2, shell=True, check=True)
            subprocess.run(command3, shell=True, check=True)

            # Combine results
            all_results.extend(
                [
                    f"{temp_dir}/result_{i}_with_region.csv",
                    f"{temp_dir}/result_{i}_without_region.csv",
                    f"{temp_dir}/result_{i}_pairwise.csv",
                ]
            )

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
        for i in range(0, len(all_results), 2):
            with open(all_results[i], "r") as infile:
                next(infile)  # skip header
                outfile1.write(infile.read())

        # Write results from second calls to second file
        for i in range(1, len(all_results), 2):
            with open(all_results[i], "r") as infile:
                next(infile)  # skip header
                outfile2.write(infile.read())
                
        # Write results from second calls to third file
        for i in range(1, len(all_results), 3):
            with open(all_results[i], "r") as infile:
                next(infile)  # skip header
                outfile3.write(infile.read())

    # Clean up temporary files
    import shutil

    shutil.rmtree(temp_dir)


def main():
    sequence_file = "sequence.fasta"
    target_file = "sequence.fasta"
    query_file = "queries.fasta"
    unpaired_prob_file = "HPV16_lunp"
    parameter_file = "parameters.cfg"

    start_time = time.time()
    print(f"Start time: {time.ctime(start_time)}")

    run_rnaplfold(sequence_file)

    with open(target_file, "r") as file:
        lines = file.readlines()
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

    motifs = find_motifs(sequence)
    queries = prepare_sequences(sequence, motifs)

    query_file = "output_queries.fasta"
    write_queries_to_fasta(queries, query_file)

    process_intarna_queries(target_file, query_file, unpaired_prob_file, parameter_file)

    end_time = time.time()
    print(f"End time: {time.ctime(end_time)}")
    print(f"Total run time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
