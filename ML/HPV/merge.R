# install.packages("tidyverse")
# install.packages(c("dplyr", "tidyr", "dbplyr", "dtplyr"))
# install.packages("conflicted")
library(conflicted)
library(tidyverse)
# set script location as working directory using Rstudio API
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

dataRootFolder <- "../../Algorithm/HPV"

# unpaired prob data
pu <-
  read_delim(str_c(dataRootFolder,"/converted_sequence_lunp"), delim="\t", skip =2, col_names = F)

pu <- set_names(pu, c( "i", str_c("l", 1:(ncol(pu)-1))))

# experimental data
ex <-
  read_csv(str_c(dataRootFolder,"/Article/data.csv")) |>
  mutate( seq = str_to_upper(Sequence) |> str_replace_all("T", "U")) |>
  mutate( seqRC = str_to_upper(SequenceRC) |> str_replace_all("T", "U"))
out <- list()
# loading each individual output file
for (i in 1:3) {
  out[[i]] <-
    read_delim( str_c(dataRootFolder,"/output/1_output",i,".csv"),
                delim=",") |>
    group_by(id2) |>
    mutate(solIdx = row_number()) |>
    ungroup() |>
    # TODO: add additional derived data columns
    #....
    # number of seeds
    mutate(
      seedNumber = str_count(seedE, ":")+1,
      seedEbest = str_extract(seedE,"^[^:]+") %>% as.numeric()
    ) |>

    # final step: add suffiy to column names
    rename_with( everything(), .fn = str_c, "_", i)
}
View(out[[3]])
# prepare NA handling
naDefaults <-
  list(
    # derived features
    seedEbest=0,
    seedNumber=0,
    # # strings and positions: NA
    # id2,seq2,
    # subseqDB,hybridDB,
    # seedStart1,seedEnd1,seedStart2,seedEnd2,
    # energies, probabilities, partition functions: default = 0
    E=0,
    Etotal=0,
    Pu1=0,
    Pu2=0,
    E_hybrid=0,
    E_norm=0,
    E_hybridNorm=0,
    Eall=0,
    Eall1=0,
    Eall2=0,
    Zall=0,
    Zall1=0,
    Zall2=0,
    EallTotal=0,
    P_E=0,
    # default == infinity
    ED1=999999,
    ED2=999999,
    # multi-value elements: string versions
    seedE="0",
    seedPu1="0",
    seedPu2="0",
    seedED1="999999",
    seedED2="999999"
  )
# merge all data sets
mergedData <-
  full_join(out[[1]], out[[2]], by =c("id2_1"="id2_2",
                                      "seq2_1"="seq2_2",
                                      "solIdx_1"="solIdx_2")) |>
  full_join(out[[3]], by =c("id2_1"="id2_3",
                            "seq2_1"="seq2_3",
                            "solIdx_1"="solIdx_3")) |>
  rename( id2 = id2_1, seq2 = seq2_1, solIdx = solIdx_1) |>
  # NA handling
  replace_na(
    # list(
    # unlist(
    c(
      set_names(naDefaults, ~str_c(.,"_1")),
      set_names(naDefaults, ~str_c(.,"_2")),
      set_names(naDefaults, ~str_c(.,"_3"))
    )
    # )
    # )
  )

# full merged data: for later exploration and prediction
write_csv(mergedData, "mergedData.csv")

# Perform the join
annotated <- left_join(mergedData, ex, by = c("seq2" = "seq"))

# Inspect before drop_na()
head(annotated)

# If data exists, apply drop_na
if(nrow(annotated) > 0) {
  annotated <- drop_na(annotated, DNAzyme)
} else {
  print("No data to process after join!")
}

# Check final result
head(annotated)
write.csv(annotated, "mergedData_annotated.csv")
