# install.packages("tidyverse")
library(tidyverse)

# set script location as working directory using Rstudio API
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

dataRootFolder <- "../../Algorithm/SARS-CoV-2"
dataFileEx <- "/Article/Fig3-data.csv"
dataFileExCol <- c( Y = "Y10" )
dataFileIds <- 1:20
dataYthreshold <- 0.14

source("../merge-functions.R")
