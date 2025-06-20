# Import package
# devtools::install_github("achambaz/tlride/tlrider")
# Import other packages
library(tidyverse)
library(caret)
library(ggdag)
library(tlrider)

# Take a look inside the downloaded packages
example(tlrider)
ls()

# Take a look at the "experiment" object
experiment

# Take a small sample from "experiment"
(five_obs <- sample_from(experiment, n = 5))
