# install.packages("tidyverse")
library(tidyverse)

# set script location as working directory using Rstudio API
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

dataRootFolder <- "../../Algorithm/Human-BCl-mRNA"
dataFileEx <- "/Article/data.csv"
dataFileExCol <- c( Y = "Y" )
dataFileIds <- 1:5
dataYthreshold <- 0.14

source("../merge-functions.R")
