# CleaveRNA

CleaveRNA is a machine learning based computational tool for scoring candidate DNAzyme cleavage sites in substrate sequences.
 
# Documentation

## Overview

## Dependencies

The following tools need to be present in the environment, e.g. via a respective coenda setup

- Python v3 (python)
- IntaRNA (intarna)
- RNAplfold (viennarna)

## Usage and parameters

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

- Feature generation contains four modes:
     - **Default**: In this mode, first the DNAzyme sequences are designed based on the given parameters then the feature table is generated for 
       all the candidate cleavage sites on the target.
       
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
         - **CS**: The cleavage sites dinucleotide of DNAzyme.
         - **Tem**: Temperature of the DNAzyme reaction.
         - **CA**: Catalytic core sequence of the DNAzyme.

      
       - If you want to select the SARS-CoV-2 model, just save it in CSV format.
          (using `--feature_mode default --params SARS_default.csv` )
         
       #### 📎 Copyable HPV-BCL (HPBC) default parameter file
       ```csv
       LA,RA,CS,Tem,CA
       9,8,"AU,GU,AC,GC",37,ggcuagcuacaacga
       ```      
       #### 📎 Copyable SARS-CoV-2 (SARS) default parameter file
       ```csv
       LA,RA,CS,Tem,CA
       16,7,"AU,GU",23,ggcuagcuacaacga
       ```
     - **Default**: In this mode, first the DNAzyme sequences are designed based on the given parameters then the feature table is generated for 
       all the candidate c
