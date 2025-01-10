

library(tidyverse)

# set script location as working directory using Rstudio API
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


ex <- read_csv("../Article/data.csv")

naDefaults <-
  list(
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




out <- list()

for (i in 1:3) {
  out[[i]] <-
  read_delim( str_c("1_output",i,".csv"),
                   delim=",") |>
  group_by(id2) |>
    mutate(solIdx = row_number()) |>
  rename_with( everything(), .fn = str_c, "_", i)
}

View(out[[1]])

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
  ) |>
  View()

write_csv(mergedData, "mergedData.csv")


