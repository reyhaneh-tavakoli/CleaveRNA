import subprocess
import re
import csv
import os
import time

# Step 1: Calculate unpaired probabilities using RNAplfold
def run_rnaplfold(sequence_file):
    command = f"RNAplfold -W 150 -L 100 -u 25 < {sequence_file}"
    subprocess.run(command, shell=True)

# Step 2: Parse RNAplfold output and find GC and AC motifs
def parse_rnaplfold_output(output_file):
    with open(output_file, 'r') as file:
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

# Step 3: Find AC and GC motifs in the target sequence
def find_motifs(sequence):
    motifs = []
    for match in re.finditer(r'AC|GC', sequence):
        start = match.start()
        end = match.end()
        if match.group() == 'AC':
            start_region = max(0, start - 10)
            end_region = min(len(sequence), end + 10)
        elif match.group() == 'GC':
            start_region = max(0, start - 10)
            end_region = min(len(sequence), end + 10)
        motifs.append((start_region, end, match.group()))
    return motifs

# Step 4: Prepare sequences for each cleavage site
def prepare_sequences(sequence, motifs):
    linker = "GGCTAGCTACAACGA"
    queries = []
    valid_nucleotides = set("ATCG")
    sequence = sequence.replace('U', 'T')  # Convert U to T
    for i, (start, end, motif) in enumerate(motifs):
        left_arm = sequence[max(0, start - 10):start]
        right_arm = sequence[end:end + 10]
        if not set(left_arm).issubset(valid_nucleotides) or not set(right_arm).issubset(valid_nucleotides):
            print(f"Skipping invalid sequence at motif {motif} with left arm {left_arm} and right arm {right_arm}")
            continue
        complementary_left_arm = ''.join(['ATCG'['TAGC'.index(n)] for n in left_arm[::-1]])
        complementary_right_arm = ''.join(['ATCG'['TAGC'.index(n)] for n in right_arm[::-1]])
        query_sequence = f"{complementary_right_arm}{linker}{complementary_left_arm}"
        # Updated query naming to include both nucleotide positions of the cleavage site
        query_name = f"{motif}-{start+10}-{start+11}"  # Includes both nucleotide positions
        queries.append((query_name, query_sequence))
    return queries# Step 5: Write sequences to a FASTA file
def write_queries_to_fasta(queries, fasta_file):
    with open(fasta_file, 'w') as f:
        for name, seq in queries:
            f.write(f">{name}\n{seq}\n")

# Step 6: Run IntaRNA for each query
def run_intarna_for_queries(target_file, query_file, unpaired_prob_file, output_file, sequence):
    motifs = find_motifs(sequence)
    queries = prepare_sequences(sequence, motifs)
    write_queries_to_fasta(queries, query_file)
    header_written = False

    for start_region, end_region, _ in motifs:
        # Ignore regions within the first 10 nucleotides and the last 10 nucleotides of the target sequence
        if start_region < 10 or end_region > len(sequence) - 10:
            print(f"Skipping region {start_region}-{end_region} as it is within the first or last 10 nucleotides of the target sequence.")
            continue

        command = (
            f"IntaRNA --target {target_file} --query {query_file} "
            f"--tAcc P --tAccFile {unpaired_prob_file} --noSeed 0 "
            f"--SeedBP 5 --intLenMax 25 --qIntLenMax 0 --mode M --MODEL X --acc C --accW 150 "
            f"--accL 100 -e V --temperature 37 --outMode C "
            f"--outNumber 2 --outOverlap N --outCsvCols id2,seq2,E,Etotal,ED1,ED2,Pu1,Pu2,subseqDB,hybridDB,"
            f"Pu2,E_dangleL,E_dangleR,E_endL,E_endR,E_init,E_loops,E_hybrid,"
            f"E_norm,E_hybridNorm,E_add,seedStart1,seedEnd1,seedStart2,seedEnd2,"
            f"seedE,seedED1,seedED2,seedPu1,Eall2,Zall,Zall1,Zall2,EallTotal,"
            f"seedPu2,w,Eall,Eall1,P_E,RT"
        )

        # Check if files exist
        if not os.path.exists(target_file):
            print(f"Error: Target file {target_file} does not exist.")
            continue
        if not os.path.exists(query_file):
            print(f"Error: Query file {query_file} does not exist.")
            continue
        if not os.path.exists(unpaired_prob_file):
            print(f"Error: Unpaired probability file {unpaired_prob_file} does not exist.")
            continue

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"IntaRNA command failed for target region {target_region} with return code {result.returncode}")
            print(f"Error message: {result.stderr}")
        else:
            lines = result.stdout.strip().split('\n')
            if not header_written:
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(lines[0].split(','))  # Write header
                    writer.writerows([line.split(',') for line in lines[1:]])
                header_written = True
            else:
                with open(output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([line.split(',') for line in lines[1:]])

# Main script
def main():
    sequence_file = 'sequence.fasta'
    target_file = 'sequence.fasta'
    query_file = 'queries.fasta'
    unpaired_prob_file = 'HPV16_lunp'
    output_file = 'output-run2-1.csv'

    start_time = time.time()
    print(f"Start time: {time.ctime(start_time)}")

    run_rnaplfold(sequence_file)
    unpaired_probs = parse_rnaplfold_output(unpaired_prob_file)

    with open(target_file, 'r') as file:
        sequence = file.read().replace('\n', '')

    run_intarna_for_queries(target_file, query_file, unpaired_prob_file, output_file, sequence)

    end_time = time.time()
    print(f"End time: {time.ctime(end_time)}")
    print(f"Total run time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()