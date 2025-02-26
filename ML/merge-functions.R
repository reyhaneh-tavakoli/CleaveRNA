
######################################
## PRELIMINARIES
##''''''''''''''''''''''''''''''''''''
## set
## - working directory
## set variables
## - dataRootFolder
## - dataFileEx = location of data file with experimental data wrt. dataRootFolder
## - dataFileExCol = c( Y = "ColNameOfExpData" )
## - dataFileIds = file prefix numbers
######################################
# install.packages("tidyverse")
library(tidyverse)

# # # set script location as working directory using Rstudio API
#setwd("/home/reyhaneh/Documents/git/RNAcutter/ML")
#getwd()

#dataRootFolder <- "../Algorithm/SARS-CoV-2"
#dataFileEx <- "/Article/Fig3-data.csv"
#dataFileExCol <- c( Y = "Y10" )
#dataFileIds <- 1:20


# experimental data
ex <-
  read_csv(str_c(dataRootFolder,dataFileEx)) |>
  mutate( seq = str_to_upper(Sequence) |> str_replace_all("T", "U"))


allMergedData <- list()
allMergedDataAnnotated <- list()

# input file specific reading
#########################################################
# Initialize lists to store data
allMergedData <- list()
allMergedDataAnnotated <- list()

for (inputFile in dataFileIds) { # modified to loop through files 1-20
  #########################################################

  # unpaired prob data
  pu <-
    read_delim(str_c(dataRootFolder,"/",inputFile, "_converted_sequence_lunp"), delim="\t", skip =2, col_names = F)
  pu <- set_names(pu, c( "i", str_c("l", 1:(ncol(pu)-1))))

  # read and process data for inputFile
  out <- list()
  # loading each individual output file
  for (i in 1:3) {
    out[[i]] <-
      read_delim( str_c(dataRootFolder,"/output/",inputFile,"_output",i,".csv"),
                  delim=",") |>
      group_by(id2) |>
      # mutate(solIdx = row_number()) |>
      slice_head(n=1) |> # keep only first solution per sequence pair
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

  # prepare NA handling
  naDefaults <-
    list(
      # derived features
      seedEbest=0,
      seedNumber=0,
      # energies, probabilities, partition functions: default = 0
      E=0,
      Pu1=0,
      Pu2=0,
      E_hybrid=0,
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
                                        "seq2_1"="seq2_2"
                                        # ,"solIdx_1"="solIdx_2"
    )) |>
    full_join(out[[3]], by =c("id2_1"="id2_3",
                              "seq2_1"="seq2_3"
                              # ,"solIdx_1"="solIdx_3"
    )) |>
    rename( id2 = id2_1
            , seq2 = seq2_1
            # , solIdx = solIdx_1
    ) |>
    # NA handling
    replace_na(
      c(
        set_names(naDefaults, ~str_c(.,"_1")),
        set_names(naDefaults, ~str_c(.,"_2")),
        set_names(naDefaults, ~str_c(.,"_3"))
      )
    ) |>
    # ensure we have a target-constraint prediction
    filter( E_1 != 0 ) |>
    # add relative differences between predictions
    mutate(
      E_diff_12 = E_2 - E_1,
    ) |>
    # add unpaired probability data features
    mutate( pos = str_extract(id2, "\\d+") |> as.numeric(),
            #         # upstream range pu
            #         pu1_4u = pull(pu,"l4")[pos-1],
            #         pu5_8u = pull(pu,"l4")[pos-5],
            #         # downstream
            #         pu1_4d = pull(pu,"l4")[pos+1+4],
            #         pu5_8d = pull(pu,"l4")[pos+1+8],
    ) |>
    rowwise() |>
    mutate(
      # upstream min of single-pos pu
      pumin1_4u = min(pull(pu,"l1")[pos-(1:4)]),
      pumin5_8u = min(pull(pu,"l1")[pos-(5:8)]),
      # downstream
      pumin1_4d = min(pull(pu,"l1")[pos+1+(1:4)]),
      pumin5_8d = min(pull(pu,"l1")[pos+1+(5:8)]),
    ) |>
    ungroup() |>
    # drop temporary column
    select( - pos ) |>
    # drop ED._1 columns, since redundant with Pu values
    select( - matches("ED[12]_1$") ) |>
    # drop all "_2" and "P_E" columns
    select( - ends_with("_2"), - starts_with("P_E_") ) |>
    # drop "_3" columns except E_3 and seedNumber_3
    select( - matches("[12ltd]_3$|Pu2_3$|(start|end).*_3$|DB_3$") )

  # Store merged data in list
  allMergedData[[as.character(inputFile)]] <- mergedData

  # Perform the join
  annotated <-
    left_join(mergedData, ex, by = c("seq2" = "seq")) |>
    # Apply drop_na
    drop_na(DNAzyme)


  # Store annotated data in list
  allMergedDataAnnotated[[as.character(inputFile)]] <- annotated

} # end of inputFile specific stuff

# write combined data
allMergedData |>
  bind_rows() |>
  distinct() |>
  write_csv("mergedData.csv")

allMergedDataAnnotated |>
  bind_rows() |>
  distinct() |>
  write_csv("mergedData_annotated.csv")


# prepare feature testing with numeric data only
numericData <-
  allMergedDataAnnotated |>
  bind_rows() |>
  # select all numeric columns
  select(where(is.numeric)) |>
  select(
    # -solIdx,
    -starts_with("P_E")) |>
  # rename target column to Y
  rename(all_of(dataFileExCol)) |>
  distinct()

# write numeric data
numericData |>
  write_csv("mergedData_annotated.num.csv")

# write balanced data
numericData |>
  # sarscov data tresholds
  filter(Y >= 0.18  | Y < 0.01 ) |> # balanced data
  write_csv("mergedData_annotated.balanced1.num.csv")

numericData |>
  # sarscov data tresholds
  filter(Y >= 0.13  | Y <= 0.01 ) |> # balanced data
  write_csv("mergedData_annotated.balanced2.num.csv")





