#!/usr/bin/env Rscript
# predict_DNAzymes_multi.R
#
# Author: Reyhaneh Tavakoli Koopaei
# Updated version: ensures consistent energy sign, proper model thresholding,
# and prevents mismatch between prob_multiple and pred_multiple.

suppressPackageStartupMessages({
  library(tools)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript predict_DNAzymes_multi.R fasta_dir output_prefix [--perl path] [--train_table path] [--dna_par path]\n")
}

fasta_dir <- args[1]
out_prefix <- args[2]

# --- Default paths ---
perl_script <- "DNAzyme_scan.pl"
train_table <- NULL
dna_par <- "dna_mathews2004.par"
tmp_dir <- tempdir()

# --- Parse optional arguments ---
if (length(args) > 2) {
  rest <- args[-(1:2)]
  i <- 1
  while (i <= length(rest)) {
    if (rest[i] == "--perl") {
      perl_script <- rest[i + 1]; i <- i + 2
    } else if (rest[i] == "--train_table") {
      train_table <- rest[i + 1]; i <- i + 2
    } else if (rest[i] == "--dna_par") {
      dna_par <- rest[i + 1]; i <- i + 2
    } else {
      stop(paste("Unknown option:", rest[i]))
    }
  }
}

cat("FASTA directory:", fasta_dir, "\n")
cat("Output prefix:", out_prefix, "\n")
cat("Perl script:", perl_script, "\n")
cat("DNA params:", dna_par, "\n")
cat("Train table:", train_table, "\n\n")

# --- Tool checks ---
check_tool <- function(cmd) nchar(Sys.which(cmd)) > 0
if (!check_tool("perl")) stop("Perl not found on PATH.")
if (!file.exists(perl_script)) stop("Perl script not found.")
if (!check_tool("RNAfold") || !check_tool("RNAduplex"))
  stop("ViennaRNA tools (RNAfold/RNAduplex) not found on PATH.")

# --- Train logistic models ---
if (is.null(train_table) || !file.exists(train_table))
  stop("Provide valid training table.")

cat("Training models from:", train_table, "\n")

DZ <- read.table(train_table, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

# Ensure numeric energies (negative values are expected)
DZ$Energy   <- as.numeric(DZ$Energy)
DZ$Dimer    <- as.numeric(DZ$Dimer)
DZ$Internal <- as.numeric(DZ$Internal)

# Response variables
DZ$good20 <- ifelse(DZ$time60 >= 20, 1, 0)
DZ$good40 <- ifelse(DZ$time60 >= 40, 1, 0)

fit_single <- glm(good20 ~ Energy, data = DZ, family = binomial(link = "logit"))
fit_multiple <- glm(good40 ~ Energy + Dimer + Internal,
                    data = DZ, family = binomial(link = "logit"))

cat("Models trained successfully.\n\n")

# --- Helper functions ---
compute_dimer <- function(seq) {
  f <- tempfile(tmpdir = tmp_dir)
  cat(">A\n", seq, "\n>B\n", seq, "\n", sep = "", file = f)
  cmd <- if (file.exists(dna_par)) {
    sprintf("RNAduplex -P %s --noGU --noTetra --noconv < %s", shQuote(dna_par), shQuote(f))
  } else {
    sprintf("RNAduplex --noconv < %s", shQuote(f))
  }
  out <- try(system(cmd, intern = TRUE), silent = TRUE)
  unlink(f)
  if (inherits(out, "try-error") || length(out) == 0) return(NA)
  num <- regmatches(out, regexpr("-?\\d+\\.?\\d*", out))
  if (length(num) == 0) return(NA)
  return(as.numeric(num[1]))
}

compute_internal <- function(seq) {
  f <- tempfile(tmpdir = tmp_dir)
  cat(">seq\n", seq, "\n", sep = "", file = f)
  cmd <- if (file.exists(dna_par)) {
    sprintf("RNAfold -p -d2 --noLP -P %s --noconv < %s", shQuote(dna_par), shQuote(f))
  } else {
    sprintf("RNAfold -p -d2 --noLP --noconv < %s", shQuote(f))
  }
  out <- try(system(cmd, intern = TRUE), silent = TRUE)
  unlink(f)
  if (inherits(out, "try-error") || length(out) == 0) return(NA)
  num <- regmatches(out, regexpr("-?\\d+\\.?\\d*", out))
  if (length(num) == 0) return(NA)
  return(as.numeric(num[1]))
}

# --- Process FASTA files ---
fasta_files <- list.files(fasta_dir, pattern = "\\.fa$|\\.fasta$", full.names = TRUE, ignore.case = TRUE)
if (length(fasta_files) == 0) stop("No FASTA files found in directory.")

for (fasta in fasta_files) {
  cat("\nProcessing:", fasta, "\n")
  fasta_name <- file_path_sans_ext(basename(fasta))
  candidates_out <- paste0(out_prefix, "_", fasta_name, "_candidates.tab")
  
  # Run Perl scanner
  perl_cmd <- sprintf("perl %s -i %s -o %s", shQuote(perl_script), shQuote(fasta), shQuote(candidates_out))
  cat("Running Perl scanner:\n", perl_cmd, "\n")
  system(perl_cmd, intern = TRUE)
  
  if (!file.exists(candidates_out) || file.info(candidates_out)$size == 0) {
    cat("No DNAzyme candidates found for", fasta, "\n")
    next
  }
  
  cand <- read.table(candidates_out, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                     quote = "", comment.char = "", fill = TRUE)
  
  if (nrow(cand) == 0) {
    cat("No DNAzyme candidates found in file.\n")
    next
  }
  
  names(cand) <- make.names(names(cand))
  if (!"DeltaG" %in% names(cand) && ncol(cand) >= 6)
    names(cand)[ncol(cand)] <- "DeltaG"
  
  # --- Feature calculation ---
  cand$Energy <- as.numeric(cand$DeltaG)
  cand$Dimer <- sapply(toupper(gsub("T", "U", gsub("[^ACGTUacgtu]", "", cand$DNAzyme))), compute_dimer)
  cand$Internal <- sapply(toupper(gsub("T", "U", gsub("[^ACGTUacgtu]", "", cand$DNAzyme))), compute_internal)
  
  # Remove NA rows before prediction
  cand <- cand[complete.cases(cand[, c("Energy", "Dimer", "Internal")]), ]
  
  # --- Prediction ---
  cand$prob_single <- predict(fit_single, newdata = cand, type = "response")
  cand$prob_multiple <- predict(fit_multiple, newdata = cand, type = "response")
  
  # Use 0.5 threshold; ensure numeric type
  cand$pred_single <- ifelse(as.numeric(cand$prob_single) > 0.5, 1, 0)
  cand$pred_multiple <- ifelse(as.numeric(cand$prob_multiple) > 0.5, 1, 0)
  
  cat("Summary of multiple model predictions:\n")
  print(table(cand$pred_multiple))
  
  # --- Save ---
  pred_file <- paste0(out_prefix, "_", fasta_name, "_predictions.csv")
  write.csv(cand, file = pred_file, row.names = FALSE, na = "")
  cat("Predictions saved to", pred_file, "\n")
}

cat("\nAll FASTA files processed successfully.\n")
