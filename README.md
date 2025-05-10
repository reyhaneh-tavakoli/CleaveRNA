# CleaveRNA

CleaveRNA is a machine learning based computational tool  for scoring candidate DNAzyme target sites in RNA sequences.
 
# Documentation

## Overview

## Dependencies

The following tools need to be present in the environment, e.g. via a respective conda setup

- Python v3 (python)
- IntaRNA (intarna)
- RNAplfold (viennarna)

## Usage and parameters

### Default mode 

CleaveRNA contains two main modules that users can select based on the purpose of evaluation. These modules are related to the feature file and train file generation.
If you just want to start and are fine with the default modes,you have to provide two files as input: 

- The target sequence in FASTA format.
```[bash]
>SARS_CoV_2_ORF1ab
TCAAGGGTACACACCACTGGTTGTTACTCACAATTTTGACTTCACTTTTAGTTTTAGTCCAGAGTACTCAATGGTCTTTGTTCTTTTTTTTGTATGAAAATGCCTTTTTACCTTTTGCTATGGGTATTATTGCTATGTCTGCTTTTGCAATGATGTTTGTCAAACATAAGCATGCATTTCTCTGTTTGTTTTTGTTACCTTCTCTTGCCACTGTAGCTTATTTTAATATGGTCTATATGCCTGCTAGTTGGGTGATGCGTATTATGACATGGTTGGATATGGTTGATACTAGTTTGTCTGGTTTTAAGCTAAAAGACTGTGTTATGTATGCATCAGCTGTAGTGTTACTAATCCTTATGACAGCAAGAACTGTGTATGATGATGGTGCTAGGAGAGTGTGGACACTTATGAATGTCTTGACACTCGTTTATAAAGTTTATTATGGTAATGCTTTAGATCAAGCCATTTCCATGTGGGCTCTTATAATCTC
```
#### Note: The minimum length of this file must be 150 nt. 

- The parameter file in CSV format.
 ### 📎 Copyable CSV Format

```csv
LA,RA,CS,temperature,core
10,15,AC,GC,37,ggcuagcuacaacga
```
#### 📊 Data Table (Formatted View)

| LA | RA | CS     | Tem         | CA                |
|----|----|--------|-------------|-------------------|
| 10 | 15 | AC,GC  | 37          | ggcuagcuacaacga   |

---
- Column Definitions
     - **LA**: Left binding arm length of the DNAzyme.
     - **RA**: Right binding arm length of the DNAzyme.
     - **Tem**: Temperature of the DNAzyme reaction.
     - **CA**: Catalytic core sequence of the DNAzyme.


- After your input files are ready run this script in the terminal
 ```[bash]
python3 CleaveRNA.py --targets target.fasta --params parameters.csv --mode_feature default --default_train_file HPBC
 ```



, and one parameter file
a (long) target RNA (using `-t` or `--target`) and a (short) query RNA
(via `-q` or `--query`), in
The CleaveRNA can be run in the default mode 
```[bash]
## Data sets

in the end we have the following data sets:

- `LA=16, RA=7, T=23` : SARS
- `LA=9, RA=8, T=37` : HPV, BCL
  
