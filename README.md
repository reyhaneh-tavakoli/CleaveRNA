# CleaveRNA

CleaveRNA is a machine learning based computational tool for scoring candidate DNAzyme cleavage sites in substrate sequences.
 
# Documentation

## Overview

## Dependencies

The following tools need to be present in the environment, e.g. via a respective coenda setup

- Python v3 (python)
- IntaRNA (intarna)
- RNAplfold (viennarna)

## Usage and Parameters

The **CleaveRNA** algorithm provides two different modes: **training** and **prediction**.

- **Training Mode**: Allows you to create a training file using your own experimental data. If you have experimental data on the fraction of DNAzyme cleavage at different target sites, you can use this mode to generate a custom training file for prediction.  
- **Prediction Mode**: Using either your own training file or the provided default ones, you can score cleavage sites on target sequences and select the most suitable DNAzyme based on your specific needs.

If you don’t have experimental data, you can use the **default training files** we provide. These were generated from experimental data published prior to the development of this algorithm.

---

### Training Mode

This section explains how the default training files were generated and how you can create your own training set for use in prediction mode.

If you have your own dataset (see details in the **`data_preparation`** folder), you must first run this mode to obtain the **pre_train** file.  

#### Steps:

1. **Prepare the target sequence files in FASTA format**  
   - Example test file: Example test files: [`BCL_1.fasta`, `BCL_2.fasta`, `BCL_3.fasta`, `BCL_4.fasta`, `BCL_5.fasta`, `HPV.fasta`](https://github.com/reytakop/CleaveRNA/tree/main/CleaveRNA/Train_mode/HPBC) 
   - **Note**:  
     - The minimum sequence length must be **150 nt**.  
     - The sequence name must match the FASTA file name.  
       - Example: The file `BCL_1.fasta` must start with `>BCL_1`.  

2. **Prepare the parameter file**  
   - Example: [`test_default.csv`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/test_default.csv)  
   - This file contains **five columns**, which are defined below:  
     - **LA**: Left binding arm length of the DNAzyme.  
     - **RA**: Right binding arm length of the DNAzyme.  
     - **CS**: The cleavage site dinucleotide of the DNAzyme. In this example, the catalytic core of the **10-23 DNAzyme** is used. [Reference: Nat. Chem. 2021](https://doi.org/10.4103/1673-5374.335157)  
     - **Tem**: Reaction temperature of the DNAzyme.  
     - **CA**: Catalytic core sequence of the DNAzyme.  

3. **Run the shell script**  
   - Script: [`run.sh`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/run)  
   - Update **lines 3–12** according to your conda environment.  
   - In the input files directory, run the tool with:  

     ```bash
     bash run
     ```

---

### Output

The tool will generate the **pre_train file**:  
[HPBC_user_merged_num.csv](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/HPBC_user_merged_num.csv)  

- In this file, you can find the generated DNAzymes (`seq_2` column) based on your defined parameters.  
- All dinucleotide cleavage sites (`id2` column) are included, along with the generated feature sets for each cleavage site.  








### Default mode 
If you just want to start and are fine with the default parameters set, you have to provide the target sequences (using `--targets`), the parameter file for feature generation (using `--param`), the default pre-train and the target variable files (using `--default_train_file` to give the prefix of files as model-name). 
```[bash]
python3 CleaveRNA.py --targets target.fasta --params test_default.csv --feature_mode default --default_train_file HPBC
```
- The target sequence in FASTA format:
```[bash]
>SARS_CoV_2_ORF1ab
TCAAGGGTACACACCACTGGTTGTTACTCACAATTTTGACTTCACTTTTAGTTTTAGTCCAGAGTACTCAATGGTCTTTGTTCTTTTTTTTGTATGAAAATGCCTTTTTACCTTTTGCTATGGGTATTATTGCTATGTCTGCTTTTGCAATGATGTTTGTCAAACATAAGCATGCATTTCTCTGTTTGTTTTTGTTACCTTCTCTTGCCACTGTAGCTTATTTTAATATGGTCTATATGCCTGCTAGTTGGGTGATGCGTATTATGACATGGTTGGATATGGTTGATACTAGTTTGTCTGGTTTTAAGCTAAAAGACTGTGTTATGTATGCATCAGCTGTAGTGTTACTAATCCTTATGACAGCAAGAACTGTGTATGATGATGGTGCTAGGAGAGTGTGGACACTTATGAATGTCTTGACACTCGTTTATAAAGTTTATTATGGTAATGCTTTAGATCAAGCCATTTCCATGTGGGCTCTTATAATCTC
```
#### Note: The minimum length of this file must be 150 nt. 

- The CleaveRNA tool contains two main modules that users can select based on the purpose of evaluation. These modules are related to both feature file and train file generation. 

- Feature generation contains four modes: default, target_screen, target_check and specific_query.

     - **Default mode**: In this mode, first the DNAzyme sequences are designed based on the given parameters then the feature table is generated 
       for all the candidate cleavage sites on the target.
       
       - You need to provide the parameter file in CSV format and using this command line option
       ```[bash]
       --feature_mode default --params test_default.csv
       ```
       - Example of parameter file:
       - We provide two different types of parameter files that are related to the SARS-CoV-2 and HPV-BCL models, respectively. You can use these 
       default parameters fileS or provide one parameter file as described in the example. 

       #### 📊 Data Table (Formatted View)

       | LA | RA | CS     | Tem         | CA                |
       |----|----|--------|-------------|-------------------|
       | 10 | 15 | AC,GC  | 37          | ggcuagcuacaacga   |
       ---
       
       - Column Definitions:
         - **LA**: Left binding arm length of the DNAzyme.
         - **RA**: Right binding arm length of the DNAzyme.
         - **CS**: The cleavage sites dinucleotide of DNAzyme. In this example the catalytic core of 10-23 DNAzyme is given.[Nat. Chem. 2021](https://doi.org/10.4103/1673-5374.335157)
         - **Tem**: Temperature of the DNAzyme reaction.
         - **CA**: Catalytic core sequence of the DNAzyme.

      
       - If you want to select the SARS-CoV-2 model , just save it in CSV format.
          (using `--feature_mode default --params SARS_default.csv` )
         
       #### 📎 Copyable HPV-BCL (HPBC) default parameter file
       ```csv
       LA,RA,CS,Tem,CA
       9,8,"AU,GU,AC,GC",37,ggcuagcuacaacga
       ```
        - This parameters extracted based on the related articles: [Nature Chem 2021](https://doi.org/10.1038/s41557-021-00645-x), [Chemistry Europe 2023](https://doi.org/10.1002/chem.202300075)
       #### 📎 Copyable SARS-CoV-2 (SARS) default parameter file
       ```csv
       LA,RA,CS,Tem,CA
       16,7,"AU,GU",23,ggcuagcuacaacga
       ```
       - This parameters extracted based on the related article: [NAR. 2023](https://doi.org/10.1002/chem.202300075)
         
     - **Target_screen mode**: In this mode, the DNAzyme sequences are designed based on the given parameters just for the cleavag sites 
       index that given, and then the feature table is generated for that region.
 
       - You need to provide the parameter file in CSV format and using this command line option
       ```[bash]
       --feature_mode target_screen --params test_target_screen.csv
       ```
       - Example of parameter file:
       - In this mode the parameter file contains one extra column (CS_index) that defines the index of each desired cleavage site.

       #### 📊 Data Table (Formatted View)
       
       | LA | RA | CS | CS_index          | Tem         | CA               |
       |----|----|----|-------------------|-------------|------------------|
       | 10 | 15 | AC | target_1.fasta:17 | 37          | ggcuagcuacaacga  |
       | 10 | 15 | CC | target_2.fasta:15 | 37          | ggcuagcuacaacga  |

       ---
       - In this parameter file, each row is the required parameter for designing spesific DNAzyme that targets the defined index on the target 
         sequnce.
         
       - Column Definition:
         - **CS_index**: The name of target file and the index of cleavage site.
         
       #### 📎 Copyable HPV-BCL (HPBC) default parameter file
       ```csv
        LA,RA,CS,CS_index,Tem,CA
        10,15,AC,1.fasta:17,37,ggcuagcuacaacga
        10,15,CC,5.fasta:15,37,ggcuagcuacaacga

       ```      
     - **Target_check mode**: In this mode, the DNAzyme sequences are designed based on the given parameters just for the cleavag sites of 
       defined target region, and then the feature table is generated for that region.
       
 
       - You need to provide the parameter file in CSV format and using this command line option
       ```[bash]
       --feature_mode target_check --params test_target_check.csv
       ```
       - Example of parameter file:
       - In this mode the parameter file contains one extra column (Start_End_Index) that defines the target region index.

       #### 📊 Data Table (Formatted View)
       
       | LA | RA | CS | Start_End_Index          | Tem         | CA               |
       |----|----|----|--------------------------|-------------|------------------|
       | 10 | 15 | AC | target_1.fasta:10-45     | 37          | ggcuagcuacaacga  |
       | 10 | 15 | AC | target_2.fasta:50-100    | 37          | ggcuagcuacaacga  |

       ---
       - In this parameter file, each row is the required parameter for designing the DNAzymes that target the defined sites of target 
         sequnce.
         
       - Column Definition:
         - **Start_End_Index**: The index of the desired region on the target site sequences. 
   
         
       #### 📎 Copyable HPV-BCL (HPBC) default parameter file
       ```csv
        LA,RA,CS,Start_End_Index,Tem,CA
        10,15,AC,1.fasta:10-45,37,ggcuagcuacaacga
        10,15,AC,5.fasta:50-100,37,ggcuagcuacaacga
       ```
     -**specific_query mode**: In this mode, the DNAzyme sequence parameters are given and the features are just generated for the them.
         - You need to provide the parameter file in CSV format and using this command line option
       ```[bash]
       --feature_mode specific_query --params test_specific_query.csv
       ```
       - Example of parameter file:
       - In this mode the parameter file contains one extra column (Start_End_Index) that defines the target region index.

       #### 📊 Data Table (Formatted View)
       
       | LA | RA | CS | Start_End_Index          | Tem         | CA               |
       |----|----|----|--------------------------|-------------|------------------|
       | 10 | 15 | AC | target_1.fasta:10-45     | 37          | ggcuagcuacaacga  |
       | 10 | 15 | AC | target_2.fasta:50-100    | 37          | ggcuagcuacaacga  |

       ---
       - In this parameter file, each row is the required parameter for designing the DNAzymes that target the defined sites of target 
         sequnce.
         
       - Column Definition:
         - **Start_End_Index**: The index of the desired region on the target site sequences. 
   
         
       #### 📎 Copyable HPV-BCL (HPBC) default parameter file
       ```csv
        LA,RA,CS,Start_End_Index,Tem,CA
        10,15,AC,1.fasta:10-45,37,ggcuagcuacaacga
        10,15,AC,5.fasta:50-100,37,ggcuagcuacaacga



