import subprocess
import pandas as pd
import os

def run_rnaplfold(sequence):
    with open("temp.fa", "w") as f:
        f.write(">temp\n" + sequence)
    
    cmd = "RNAplfold -u 10 -W 80 -L 40 < temp.fa"
    subprocess.run(cmd, shell=True, check=True)
    
    unpair_prob = []
    with open("temp_lunp", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            try:
                unpair_prob.append(float(line.split()[1]))
            except (ValueError, IndexError):
                continue
    return unpair_prob

def find_cleavage_sites(sequence):
    sites = []
    for i in range(len(sequence) - 1):
        dinucleotide = sequence[i:i+2]
        if dinucleotide in ["GC", "AC"]:
            sites.append({
                'position': i + 1,
                'sequence': dinucleotide,
                'site': i
            })
    return sites

def get_binding_arms(sequence, site, left_length=10, right_length=9):
    left_start = max(0, site - left_length)
    right_end = min(len(sequence), site + 2 + right_length)
    left_arm = sequence[left_start:site]
    right_arm = sequence[site+2:right_end]
    return left_arm, right_arm

def calculate_binding_arm_unpair_prob(unpair_prob, start, length):
    end = min(len(unpair_prob), start + length)
    return sum(unpair_prob[start:end])

def get_complementary_sequence(seq):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, base) for base in seq[::-1])

def create_query_sequences(left_df, right_df):
    queries = []
    linker = "GGCTAGCTACAACGA"
    
    left_filtered = left_df[
        (left_df['Arranged UBP of left arm'] >= 7.0) & 
        (left_df['Arranged UBP of left arm'] <= 10.0)
    ]
    
    right_filtered = right_df[
        (right_df['Arranged UBP of right arm'] >= 7.0) & 
        (right_df['Arranged UBP of right arm'] <= 10.0)
    ]
    
    for _, row in left_filtered.iterrows():
        comp_seq = get_complementary_sequence(row['left sequence'])
        queries.append({
            'arm_type': 'left',
            'cs_index': row['C.S.'],
            'query_seq': f"{comp_seq}{linker}{comp_seq}"
        })
    
    for _, row in right_filtered.iterrows():
        comp_seq = get_complementary_sequence(row['right sequence'])
        queries.append({
            'arm_type': 'right',
            'cs_index': row['C.S.'],
            'query_seq': f"{comp_seq}{linker}{comp_seq}"
        })
    
    return queries

def save_query_fastas(queries):
    for i, query in enumerate(queries, 1):
        with open(f"Q{i}.fasta", "w") as f:
            f.write(f">Query{i}\n{query['query_seq']}\n")
    return len(queries)

def run_intarna(query_file, target_file, output_file):
    cmd = f"IntaRNA -q {query_file} -t {target_file} --qAcc C --qPfScale 1 --tAcc C --tPfScale 1 --noSeed 0 --seedBP 7 -m H --model X --acc C -e V --energyVRNA Turner04 --temperature 37 --out {output_file} --outMode D"
    subprocess.run(cmd, shell=True, check=True)

def process_intarna_results(num_queries, target_file):
    results = []
    for i in range(1, num_queries + 1):
        query_file = f"Q{i}.fasta"
        csv_output = f"Q{i}.CSV"
        excel_output = f"Q{i}.xlsx"
        
        # Run IntaRNA
        run_intarna(query_file, target_file, csv_output)
        
        # Convert CSV to Excel
        df = pd.read_csv(csv_output)
        df.to_excel(excel_output, index=False)
        results.append(df)
        
        # Remove CSV file after conversion
        if os.path.exists(csv_output):
            os.remove(csv_output)
    
    return results

def create_final_output(queries, intarna_results, left_sorted, right_sorted):
    final_results = []
    
    # Create initial results with E(hybrid) values
    for i, query in enumerate(queries, 1):
        intarna_file = f"Q{i}.xlsx"
        intarna_df = pd.read_excel(intarna_file)
        
        # Extract E(hybrid) value
        e_hybrid = None
        for idx, row in intarna_df.iterrows():
            for col in intarna_df.columns:
                cell_value = str(intarna_df.iloc[idx][col])
                if 'E(hybrid)' in cell_value:
                    e_hybrid = float(cell_value.split('=')[-1].strip())
                    break
        
        # Get correct binding arm sequence
        binding_arm_seq = ""
        if query['arm_type'] == 'left':
            binding_arm_seq = left_sorted[left_sorted['C.S.'] == query['cs_index']]['left sequence'].values[0]
        else:
            binding_arm_seq = right_sorted[right_sorted['C.S.'] == query['cs_index']]['right sequence'].values[0]
            
        result = {
            'C.S. Index': query['cs_index'],
            'Binding Arm Sequence': binding_arm_seq,
            'Query Sequence': query['query_seq'],
            'E(hybrid)': e_hybrid
        }
        
        final_results.append(result)
    
    # Create DataFrame and sort all columns based on E(hybrid)
    final_df = pd.DataFrame(final_results)
    final_df = final_df.sort_values(by='E(hybrid)', ascending=True)  # Sort from most negative to least negative
    
    # Write sorted results to Excel
    with pd.ExcelWriter('rnacutter.xlsx', mode='a', if_sheet_exists='replace') as writer:
        final_df.to_excel(writer, sheet_name='Final Results', index=False)
        final_df.to_excel(writer, sheet_name='Cleavage Site Scoring', index=False)
    
    return final_df

def analyze_sequence(sequence_file):
    # Read sequence from FASTA file
    with open(sequence_file, "r") as f:
        sequence = f.read().splitlines()[1]

    # Get unpaired probabilities
    unpair_prob = run_rnaplfold(sequence)
    
    # Basic nucleotide analysis
    nuc_data = []
    for i, nucleotide in enumerate(sequence):
        nuc_data.append({
            'NUC index': i + 1,
            'Nucleotide': nucleotide,
            'U.N score': unpair_prob[i] if i < len(unpair_prob) else 0
        })
    
    # Create basic DataFrame
    basic_df = pd.DataFrame(nuc_data)
    
    # Cleavage sites analysis
    cleavage_sites = find_cleavage_sites(sequence)
    cs_data = []
    
    for site_info in cleavage_sites:
        site = site_info['site']
        left_arm, right_arm = get_binding_arms(sequence, site)
        
        left_unpair = calculate_binding_arm_unpair_prob(unpair_prob, max(0, site - 10), 10)
        right_unpair = calculate_binding_arm_unpair_prob(unpair_prob, site + 2, 9)
        
        cs_data.append({
            'C.S.': site_info['position'],
            'C.S. Nu': site_info['sequence'],
            'C.S.UN': unpair_prob[site] + unpair_prob[site + 1],
            'left binding arm seq': left_arm,
            'U.B.P of left binding arm': left_unpair,
            'right binding arm seq': right_arm,
            'U.B.P of right binding arm': right_unpair
        })
    
    cs_df = pd.DataFrame(cs_data)
    
    # Create sorted versions for left and right binding arms

    left_sorted = cs_df[['C.S.', 'U.B.P of left binding arm', 'left binding arm seq']].sort_values(
        'U.B.P of left binding arm', ascending=False
    ).rename(columns={
        'U.B.P of left binding arm': 'Arranged UBP of left arm',
        'left binding arm seq': 'left sequence'
    })
    

    right_sorted = cs_df[['C.S.', 'U.B.P of right binding arm', 'right binding arm seq']].sort_values(
        'U.B.P of right binding arm', ascending=False
    ).rename(columns={
        'U.B.P of right binding arm': 'Arranged UBP of right arm',
        'right binding arm seq': 'right sequence'
    })
    


    # Create query sequences
    queries = create_query_sequences(left_sorted, right_sorted)
    
    # Save query sequences as FASTA files
    num_queries = save_query_fastas(queries)
    
    # Run IntaRNA and process results
    intarna_results = process_intarna_results(num_queries, sequence_file)
    
    # Create final output
    final_df = create_final_output(queries, intarna_results, left_sorted, right_sorted)
    
    # Save to Excel
    with pd.ExcelWriter('rnacutter.xlsx') as writer:
        basic_df.to_excel(writer, sheet_name='Nucleotide Analysis', index=False)
        cs_df.to_excel(writer, sheet_name='Cleavage Sites', index=False)
        left_sorted.to_excel(writer, sheet_name='Sorted Left Arms', index=False)
        right_sorted.to_excel(writer, sheet_name='Sorted Right Arms', index=False)
        final_df.to_excel(writer, sheet_name='Final Results', index=False)
    






    # Cleanup
    cleanup_files(num_queries)
def cleanup_files(num_queries):
    files_to_remove = ["temp.fa", "temp_dp.ps", "temp_lunp"]
    files_to_remove.extend([f"Q{i}.fasta" for i in range(1, num_queries + 1)])
    files_to_remove.extend([f"Q{i}.xlsx" for i in range(1, num_queries + 1)])
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)

# Example usage
analyze_sequence("your_sequence.fasta")
