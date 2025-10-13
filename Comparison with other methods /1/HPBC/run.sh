#!/bin/bash
FASTA_DIR="."
OUT_PREFIX="results_HPV16"
PERL_SCRIPT="DNAzyme_scan.pl"
TRAIN_TABLE="HPV_FullLength.tab"
DNA_PAR="dna_mathews2004.par"

Rscript predict_DNAzymes.R "$FASTA_DIR" "$OUT_PREFIX" \
    --perl "$PERL_SCRIPT" \
    --train_table "$TRAIN_TABLE" \
    --dna_par "$DNA_PAR"

