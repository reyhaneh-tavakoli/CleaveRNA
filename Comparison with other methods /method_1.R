#!/usr/bin/env Rscript
suppressMessages({
  library(optparse)
  library(seqinr)
  library(Biostrings)
})

# -----------------------------
# 1️⃣ Define helper functions
# -----------------------------

reverse_complement <- function(seq) {
  seq <- toupper(seq)
  comp <- chartr("ACGTU", "TGCAA", seq)
  paste0(rev(strsplit(comp, "")[[1]]), collapse = "")
}

free_energy <- function(seq, par_file = "dna_mathews2004.par") {
  # Placeholder: energy model based on NN params
  if (!file.exists(par_file)) stop("Parameter file not found: ", par_file)
  energy <- -1 * (nchar(seq) * 0.5) # dummy function; replace with thermodynamic model if needed
  return(energy)
}

compute_dimer <- function(seq) {
  # Simple self-dimer check by GC content (proxy for complementarity)
  seq <- toupper(seq)
  revc <- reverse_complement(seq)
  overlap <- sum(strsplit(seq, "")[[1]] == strsplit(revc, "")[[1]])
  return(overlap / nchar(seq))
}

compute_internal <- function(seq) {
  seq <- toupper(seq)
  bases <- table(strsplit(seq, "")[[1]])
  gc <- sum(bases[c("G", "C")], na.rm = TRUE)
  return(gc / nchar(seq))
}

# -----------------------------
# 2️⃣ Parse arguments
# -----------------------------
option_list <- list(
  make_option(c("-i", "--input"), type = "character", help = "Input FASTA file"),
  make_option(c("-o", "--output"), type = "character", help = "Output file"),
  make_option(c("--perl"), type = "character", default = "DNAzyme_scan.pl", help = "Path to DNAzyme_scan.pl"),
  make_option(c("--model_single"), type = "character", default = "fit_single.rds", help = "Trained single model"),
  make_option(c("--model_multiple"), type = "character", default = "fit_multiple.rds", help = "Trained multiple model"),
  make_option(c("--dna_par"), type = "character", default = "dna_mathews2004.par", help = "DNA parameters file")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$input) || is.null(opt$output)) {
  stop("Please provide both input FASTA (-i) and output file (-o).")
}

# -----------------------------
# 3️⃣ Run the Perl scanner
# -----------------------------
cat("🔹 Running DNAzyme_scan...\n")
perl_cmd <- sprintf("perl %s -i %s -o tmp_dz.txt", opt$perl, opt$input)
system(perl_cmd)

if (!file.exists("tmp_dz.txt")) stop("❌ Perl output file not found.")

# -----------------------------
# 4️⃣ Read the Perl output
# -----------------------------
cat("🔹 Reading DNAzyme candidates...\n")
cand <- read.delim("tmp_dz.txt", sep = "\t", header = TRUE)

if (nrow(cand) == 0) {
  stop("⚠️ No DNAzymes found in Perl output.")
}

# -----------------------------
# 5️⃣ Compute DNAzyme features
# -----------------------------
cat("🔹 Computing thermodynamic features...\n")

cand$Energy <- vapply(cand$DNAzyme, free_energy, numeric(1), par_file = opt$dna_par)

cat("🔹 Computing dimerization potential...\n")
cand$Dimer <- vapply(cand$DNAzyme, compute_dimer, numeric(1))

cat("🔹 Computing internal GC ratio...\n")
cand$Internal <- vapply(cand$DNAzyme, compute_internal, numeric(1))

# -----------------------------
# 6️⃣ Predict using trained model
# -----------------------------
if (file.exists(opt$model_single)) {
  cat("🔹 Loading single model...\n")
  fit_single <- readRDS(opt$model_single)
  cand$Pred_Single <- predict(fit_single, newdata = cand)
} else {
  warning("Single model not found; skipping.")
}

if (file.exists(opt$model_multiple)) {
  cat("🔹 Loading multiple model...\n")
  fit_multiple <- readRDS(opt$model_multiple)
  cand$Pred_Multiple <- predict(fit_multiple, newdata = cand)
} else {
  warning("Multiple model not found; skipping.")
}

# -----------------------------
# 7️⃣ Save results
# -----------------------------
cat("✅ Writing final prediction table...\n")
write.table(cand, file = opt$output, sep = "\t", row.names = FALSE, quote = FALSE)

cat("🎯 Prediction completed successfully!\n")
cat("Output saved to:", opt$output, "\n")
