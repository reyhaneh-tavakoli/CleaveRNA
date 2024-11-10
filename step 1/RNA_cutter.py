import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import subprocess
import os

def read_and_index_fasta(fasta_file):
    sequence = str(next(SeqIO.parse(fasta_file, "fasta")).seq)
    df = pd.DataFrame({
        'NUC index': range(1, len(sequence) + 1),
        'Nucleotide': list(sequence)
    })
    return df, sequence

def calculate_unpaired_probabilities(sequence):
    with open("temp.fa", "w") as f:
        f.write(f">seq\n{sequence}")
    subprocess.run(["RNApLfold", "-W", "80", "-L", "40", "-u", "1"])
    unpaired_probs = pd.read_csv("temp_lunp", delim_whitespace=True, header=None)[2]
    return unpaired_probs

def identify_cleavage_sites(sequence, unpaired_probs):
    sites = []
    for i in range(len(sequence)-1):
        dinucleotide = sequence[i:i+2]
        if dinucleotide in ['GC', 'AC']:
            sites.append({
                'C.S.': i+1,
                'C.S. Nu': dinucleotide,
                'C.S.UN': unpaired_probs[i] + unpaired_probs[i+1]
            })
    return pd.DataFrame(sites)

def extract_binding_arms(sequence, sites_df, unpaired_probs):
    left_arms = []
    right_arms = []
    
    for _, site in sites_df.iterrows():
        pos = site['C.S.'] - 1
        
        # Left arm processing
        left_start = max(0, pos-10)
        left_seq = sequence[left_start:pos]
        left_unpaired = sum(unpaired_probs[left_start:pos])
        
        # Right arm processing
        right_start = pos + 2
        right_seq = sequence[right_start:right_start+9]
        right_unpaired = sum(unpaired_probs[right_start:right_start+9])
        
        left_arms.append({
            'C.S.': site['C.S.'],
            'left binding arm seq': left_seq,
            'U.B.P of left binding arm': left_unpaired
        })
        
        right_arms.append({
            'C.S.': site['C.S.'],
            'right binding arm seq': right_seq,
            'U.B.P of right binding arm': right_unpaired
        })
    
    return pd.DataFrame(left_arms), pd.DataFrame(right_arms)

def generate_query_sequences(left_df, right_df):
    linker = "GGCTAGCTACAACGA"
    queries = []
    
    # Process qualifying arms
    for df, arm_type in [(left_df, 'Left'), (right_df, 'Right')]:
        filtered = df[df[f'U.B.P of {arm_type.lower()} binding arm'].between(7.0, 10.0)]
        for _, row in filtered.iterrows():
            seq = row[f'{arm_type.lower()} binding arm seq']
            comp_seq = str(Seq(seq).reverse_complement())
            query = f"{comp_seq}{linker}{comp_seq}"
            queries.append({
                'C.S.': row['C.S.'],
                'Arm_Type': arm_type,
                'Query_sequence': query
            })
    
    return pd.DataFrame(queries)

def run_intarna_analysis(queries_df, target_fasta):
    os.makedirs("queries", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    all_results = []
    for idx, row in queries_df.iterrows():
        query_file = f"queries/Q{idx+1}.fasta"
        with open(query_file, "w") as f:
            f.write(f">Q{idx+1}\n{row['Query_sequence']}")
        
        output_file = f"results/Q{idx+1}.csv"
        intarna_cmd = [
            "IntaRNA",
            "-q", query_file,
            "-t", target_fasta,
            "--qAcc", "C",
            "--qPfScale", "1",
            "--tAcc", "C",
            "--tPfScale", "1",
            "--noSeed", "0",
            "--seedBP", "7",
            "-m", "H",
            "--model", "X",
            "--acc", "C",
            "-e", "V",
            "--energyVRNA", "Turner04",
            "--temperature", "37",
            "--out", output_file,
            "--outMode", "D"
        ]
        subprocess.run(intarna_cmd)
        
        intarna_results = pd.read_csv(output_file)
        all_results.append({
            'C.S.': row['C.S.'],
            'Arm_Type': row['Arm_Type'],
            'Query_sequence': row['Query_sequence'],
            'IntaRNA_results': intarna_results.to_dict('records')
        })
    
    return pd.DataFrame(all_results)

def main(fasta_file):
    print("Starting RNAcutter analysis...")
    
    # Core processing
    df, sequence = read_and_index_fasta(fasta_file)
    unpaired_probs = calculate_unpaired_probabilities(sequence)
    df['U.N score'] = unpaired_probs
    
    sites_df = identify_cleavage_sites(sequence, unpaired_probs)
    left_df, right_df = extract_binding_arms(sequence, sites_df, unpaired_probs)
    
    # Sort arms by unpaired probability
    left_df_sorted = left_df.sort_values('U.B.P of left binding arm', ascending=False)
    right_df_sorted = right_df.sort_values('U.B.P of right binding arm', ascending=False)
    
    queries_df = generate_query_sequences(left_df_sorted, right_df_sorted)
    final_results = run_intarna_analysis(queries_df, fasta_file)
    
    # Save all results to Excel
    with pd.ExcelWriter('rnacutter.excel') as writer:
        df.to_excel(writer, sheet_name='Nucleotide_Index', index=False)
        sites_df.to_excel(writer, sheet_name='Cleavage_Sites', index=False)
        left_df_sorted.to_excel(writer, sheet_name='Left_Arms', index=False)
        right_df_sorted.to_excel(writer, sheet_name='Right_Arms', index=False)
        queries_df.to_excel(writer, sheet_name='Queries', index=False)
        final_results.to_excel(writer, sheet_name='Final_Results', index=False)
    
    print("Analysis complete! Results saved to rnacutter.excel")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python RNA_cutter.py your_sequence.fasta")
        sys.exit(1)
    main(sys.argv[1])
