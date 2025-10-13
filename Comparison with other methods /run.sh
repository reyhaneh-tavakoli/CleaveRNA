chmod +x predict_DNAzymes.R

./predict_DNAzymes.R \
  -i input_HPV16.fasta \
  -o results_HPV16.tsv \
  --perl DNAzyme_scan.pl \
  --model_single fit_single.rds \
  --model_multiple fit_multiple.rds \
  --dna_par dna_mathews2004.par

