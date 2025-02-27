# install.packages("tidyverse")
library(tidyverse)

# set script location as working directory using Rstudio API
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

dataRootFolder <- "../../Algorithm/HPV"
dataFileEx <- "/Article/data.csv"
dataFileExCol <- c( Y = "Y60" )
dataFileIds <- 1
dataYthreshold <- 0.3

# run the merge functions
source("../merge-functions.R")
