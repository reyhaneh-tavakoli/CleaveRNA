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

If you have your own dataset (see details in the **`data_preparation`** folder), you must first run this mode to generate the **pre_train** file.

#### Steps

1. **Prepare the target sequence files in FASTA format**  
   - Example test files: [`BCL-1.fasta`, `BCL-2.fasta`, `BCL-3.fasta`, `BCL-4.fasta`, `BCL-5.fasta`, `HPV.fasta`](https://github.com/reytakop/CleaveRNA/tree/main/CleaveRNA/Train_mode/HPBC)  
   - **Notes:**  
     - The minimum sequence length must be **150 nt**.  
     - The sequence name must match the FASTA file name.  

       **Example:** If the target file is [`BCL-1.fasta`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/BCL-1.fasta), the header must start with:  

       ```fasta
       >BCL-1
       GTTGGCCCCCGTTACTTTTCCTCTGGGAAATATGGCGCACGCTGGGAGAACAGGGTACGATAACCGGGAG
       ATAGTGATGAAGTACATCCATTATAAGCTGTCGCAGAGGGGCTACGAGTGGGATGCGGGAGATGTGGGCG
       CCGCGCCCCCGGGGGCCGCCCCCGCGCCGGGCATCTTCTCCTCGCAGCCCGGGCACACGCCCCATACAGC
       ...
       ```
       The target files must be provided with the `--targets` flag:  
       ```bash
       --targets
       ```
       - Example: The file `BCL-1.fasta` must start with `>BCL-1`.  
2. **Prepare the parameter file (default mode)**  
   - Example: [`test_default.csv`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/test_default.csv)  
   - This file contains **five columns**, described below:  
     - **LA**: Left binding arm length of the DNAzyme  
     - **RA**: Right binding arm length of the DNAzyme  
     - **CS**: Cleavage site dinucleotide of the DNAzyme. In this example, the catalytic core of the **10-23 DNAzyme** is used ([Reference: Nat. Chem. 2021](https://doi.org/10.4103/1673-5374.335157))  
     - **Tem**: Reaction temperature of the DNAzyme  
     - **CA**: Catalytic core sequence of the DNAzyme  
   - Provide the default mode and parameter file with:  
     ```bash
     --feature_mode default --params test_default.csv
     ```
3. **Define the output directory**  
      ```bash
      --output_dir
      ```
4. **Specify the model name**  
   - Provide the model name using the `--model_name` flag:
     
     ```bash
     --model_name "HPBC"
     ```
     
5. **Run the shell script**
   
   - Script: [`run.sh`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/run)  
   - Update **lines 3–12** to match your conda environment.  
   - In the input files directory, run the tool with:
     
     ```bash
     bash run.sh
     ```
    
### Output

The tool will generate the **pre_train file**:  
[HPBC_user_merged_num.csv](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/HPBC_user_merged_num.csv)  

- In this file, you can find the generated DNAzymes (`seq_2` column) based on your defined parameters.  
- All dinucleotide cleavage sites (`id2` column) are included, along with the generated feature sets for each cleavage site.
  
**Note:**  
Two different **pre_train** files are provided, generated from the largest fraction cleavage dataset published prior to the development of this tool.  

If you have your own dataset (or a newly published one), please:  
1. Create a new folder and name it according to your `model_name`.  
2. Prepare all the required input files as described above.  
3. Update the `run.sh` script and run it. 

---

### Prediction Mode (Default)

In this mode, you can use the generated **pre_train file** to score the cleavage sites on your target files based on machine learning predictions. First, the DNAzyme sequences are designed according to the given parameters, and then all cleavage site positions are classified and scored based on the AI predictions.

The required input files are:  

1. **Target sequence FASTA files**  
   These are the sequences you want to consider as targets for DNAzyme.  
   - Example test files: [`BCL-1.fasta`, `BCL-2.fasta`, `BCL-3.fasta`, `BCL-4.fasta`, `BCL-5.fasta`, `HPV.fasta`](https://github.com/reytakop/CleaveRNA/tree/main/CleaveRNA/Prediction_mode/default/HPBC)  
   - **Notes**:  
     - The minimum sequence length must be **150 nt**.  
     - The sequence name must match the FASTA file name. For example, if the target file is [`BCL-1.fasta`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Prediction_mode/default/HPBC/BCL-1.fasta), the header must start with:  

       ```bash
       >BCL-1
       GTTGGCCCCCGTTACTTTTCCTCTGGGAAATATGGCGCACGCTGGGAGAACAGGGTACGATAACCGGGAG
       ATAGTGATGAAGTACATCCATTATAAGCTGTCGCAGAGGGGCTACGAGTGGGATGCGGGAGATGTGGGCG
       CCGCGCCCCCGGGGGCCGCCCCCGCGCCGGGCATCTTCTCCTCGCAGCCCGGGCACACGCCCCATACAGC
       ...
       ```

2. **Parameter file**  
   In this example, the default parameter mode is used.  
   - Example: [`test_default.csv`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Train_mode/HPBC/test_default.csv)  
   - This file contains **five columns** (defined previously) and must be uploaded with the following command:  

     ```bash
     --feature_mode default --params test_default.csv
     ```

3. **Pre_train file**  
   - Example: [`HPBC_user_merged_num.csv`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Prediction_mode/default/HPBC/HPBC_user_merged_num.csv)  
   - **Notes**:  
     - The pre_train file is either generated during `train_mode` or you can use the default provided file.  
     - Upload this file using:  

       ```bash
       --prediction_mode HPBC_user_merged_num.csv
       ```

4. **Model name**  
   Select the model name, which will be used as the prefix for all generated output files.  
   - Specify it with:  

     ```bash
     ----model_name HPBC
     ```

5. **Classification score file**  
   - Example: [`HPBC_target.csv`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Prediction_mode/default/HPBC/HPBC_target.csv)  
   - This file contains **two columns**:  
     - **id2** → Cleavage site index  
     - **Y** → Classification score of that position based on experimental data  
   - **Notes**:  
     - This file must be prepared by the user or you can use the corresponding file from the default training set.  
     - If using your own dataset, the fraction cleavage of each site can be converted to binary classification as described in the **`data_preparation`** folder.  
     - Upload this file with:  

       ```bash
       ----ML_target HPBC_target.csv
       ```

6. **Define the output directory**  

   ```bash
   --output_dir
   ```
      
7. **Run the shell script**

   - Script: [`run.sh`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Prediction_mode/default/HPBC/run.sh)  
   - Update **lines 3–14** in the script to match your conda environment.  
   - From the input files directory, run the tool using:  

     ```bash
     bash run.sh
     ```

### Output

The tool will generate the following output files:

1. **Model performance metrics**  
   - Example: [`HPBC_ML_metrics.csv`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Prediction_mode/default/HPBC/HPBC_ML_metrics.csv)  
   - Contains all machine learning scores related to the prediction.  

2. **Prediction file**  
   - Example: [`HPBC_CleaveRNA_output.csv`](https://github.com/reytakop/CleaveRNA/blob/main/CleaveRNA/Prediction_mode/default/HPBC/HPBC_CleaveRNA_output.csv)  
   - This file reports candidate cleavage sites scored by their accessibility for DNAzyme cleavage reactions.  

   **Columns included:**  
   - **CS_Index** → Nucleotide index of the cleavage site on the target sequence  
   - **Dz_Seq** → DNAzyme sequence designed for each cleavage site  
   - **CS_Target_File** → Target file name associated with each cleavage site  
   - **Classification_score** → Binary classification of the cleavage sites based on ML prediction  
   - **Prediction_score** → Score reflecting the accuracy of prediction at each position  
   - **Decision_score** → Model decision score.
     
 ---
 
 ### Prediction Mode (Target_screen)
 
 In this feature mode, the prediction just done for the cleavage sites that index given by user
 The input files in this mode is the same as defaulit mode except the one related to the **parameter file** that is described in detail below:  
 
  -The parameter file in this mode is the CSV file that contain six columns: 
  
 
         
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



