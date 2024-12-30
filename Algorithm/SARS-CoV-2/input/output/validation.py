import os
import pandas as pd
import argparse

def convert_t_to_u_and_rename_headers(input_file, output_file, sequence_col_index=1):
    try:
        df = pd.read_csv(input_file)
        df.iloc[:, sequence_col_index] = df.iloc[:, sequence_col_index].str.replace('T', 'U')
        
        # Get original column names
        original_columns = df.columns.tolist()
        
        # Rename only the first three columns if they exist
        if len(original_columns) >= 3:
            original_columns[0] = 'region number'
            original_columns[1] = 'sequence'
            original_columns[2] = 'position on genome'
        
        df.columns = original_columns
        df.to_csv(output_file, index=False)
        return True, original_columns
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
        return False, None

def process_rna_sequences(input_file, all_columns):
    try:
        df = pd.read_csv(input_file)
        all_results = []
        
        for index, row in df.iterrows():
            current_results = []
            
            # Create dictionary with all columns from input file
            row_dict = {col: row[col] for col in all_columns}
            current_results.append(row_dict)
            
            number = str(int(row.iloc[0]))
            sequence = str(row.iloc[1]).strip()
            
            matching_files = [f for f in os.listdir('.')
                            if f.startswith(f"{number}_") 
                            and f.endswith('.csv')
                            and f != input_file
                            and f != 'EX-RC.csv']
            
            for file in sorted(matching_files)[:3]:
                try:
                    match_df = pd.read_csv(file, sep=';')
                    matching_rows = match_df[match_df['seq2'].str.strip() == sequence]
                    
                    if not matching_rows.empty:
                        for _, match_row in matching_rows.iterrows():
                            current_results.append(match_row.to_dict())
                            print(f"Match found in {file}")
                
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
            
            if len(current_results) > 1:
                all_results.extend(current_results)
        
        if all_results:
            output_df = pd.DataFrame(all_results)
            output_df.to_csv('EX-RC.csv', sep=';', index=False)
            print(f"Total entries: {len(all_results)}")

    except Exception as e:
        print(f"Error processing sequences: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process RNA sequences from CSV file')
    parser.add_argument('input_file', help='Input CSV file name')
    parser.add_argument('--output', default='Ex.csv', help='Output CSV file name')
    args = parser.parse_args()

    success, columns = convert_t_to_u_and_rename_headers(args.input_file, args.output)
    if success:
        process_rna_sequences(args.output, columns)

if __name__ == "__main__":
    main()