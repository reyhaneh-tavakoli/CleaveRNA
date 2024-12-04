

library(tidyverse)

# set working directory to script file location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

read_csv("run1-Ex data.csv") |>
  pivot_longer(cols = c("time60","SEM60"), names_to = "exp.data", values_to = "measure") |>
  mutate(measureNo0 = ifelse(abs(measure)<0.5,NA,measure)) |>
  ggplot(aes(x = E
               # str_extract(seedE,"^[^:]*") |> as.numeric()
             , y = measureNo0)) +
  geom_point()+
  geom_smooth(aes(y=measureNo0), method = "lm", se = FALSE) +
  facet_wrap(vars(exp.data), scales = "free_y")

read_csv("Results_with_region.csv") |>
  group_by(seq2) |>
  filter(n()>1)
  View()
