import os
import pandas as pd

def convert_t_to_u_and_rename_headers(input_file, output_file):
    """
    Convert T to U in the second column and rename headers of the input CSV file.
    """
    try:
        # Read the input CSV file
        df = pd.read_csv(input_file)
        
        print(f"Converting T to U in sequences from {input_file}")
        
        # Convert T to U in the second column (index 1)
        df.iloc[:, 1] = df.iloc[:, 1].str.replace('T', 'U')
        
        # Rename columns
        df.columns = ['region number', 'sequence', 'position on genome']
        
        # Save to new CSV file
        df.to_csv(output_file, index=False)
        print(f"Converted sequences saved to {output_file} with renamed headers")
        
        return True
        
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
        return False

def process_rna_sequences():
    """
    Process RNA sequences from Ex.csv and find matches in output files.
    """
    # Read the Ex.csv file
    try:
        ex_df = pd.read_csv('Ex.csv')
        print("\nContents of Ex.csv:")
        print(ex_df)
    except FileNotFoundError:
        print("Error: Ex.csv file not found")
        return
    
    # Initialize list to store all results
    all_results = []
    
    # Process each row in Ex.csv
    for index, row in ex_df.iterrows():
        current_results = []  # Store results for current sequence
        
        number = str(int(row['region number']))  # First column value
        sequence = row['sequence'].strip()  # Second column value with whitespace removed
        position = row['position on genome']  # Third column value
        
        print(f"\nProcessing sequence #{number}: {sequence}")
        
        # First, add the row from Ex.csv with headers
        current_results.append({
            'region number': number,
            'sequence': sequence,
            'position on genome': position
        })
        
        # Find CSV files starting with the number
        matching_files = [f for f in os.listdir('.')
                        if f.startswith(f"{number}_") 
                        and f.endswith('.csv')
                        and f != 'Ex.csv'
                        and f != 'EX-RC.csv'
                        and f != 'Ex-data.csv']
        
        if not matching_files:
            print(f"No output files found for sequence #{number}")
        else:
            print(f"Found files for sequence #{number}: {sorted(matching_files)}")
        
        # Process up to 3 matching files
        for file in sorted(matching_files)[:3]:
            try:
                # Read each matching CSV file with semicolon separator
                print(f"Checking file: {file}")
                df = pd.read_csv(file, sep=';')
                
                # Find matches in seq2 column
                matching_rows = df[df['seq2'].str.strip() == sequence]
                
                # Add matching rows to results
                if not matching_rows.empty:
                    for _, match_row in matching_rows.iterrows():
                        row_dict = match_row.to_dict()
                        current_results.append(row_dict)
                        print(f"Match found in {file}")
                else:
                    print(f"No matches found in {file}")
                
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
        
        # Add all results for this sequence to the main results list
        if len(current_results) > 1:  # Only add if we found matches
            all_results.extend(current_results)
    
    # Create output DataFrame and save to CSV
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df.to_csv('EX-RC.csv', sep=';', index=False)
        print("\nResults saved to EX-RC.csv")
        print(f"Total entries: {len(all_results)}")
        print("\nFirst few rows of output file:")
        print(output_df.head())
    else:
        print("\nNo matches found")

def main():
    # First convert T to U and rename headers
    if convert_t_to_u_and_rename_headers('Ex-data.csv', 'Ex.csv'):
        print("\nStarting sequence processing...")
        # Then process the sequences
        process_rna_sequences()
    else:
        print("Error in conversion and header renaming. Stopping process.")

if __name__ == "__main__":
    main()