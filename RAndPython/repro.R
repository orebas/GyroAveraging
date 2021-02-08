library(ggplot2)
library(tidyverse)
library(plyr)
library(dplyr)
library(readr)
library(tikzDevice)


uniqFuncs <- c("Nonsmoothabs","Nonsmoothrungeabs","Nonsmoothsqrt","Smoothexp","Smoothpoly", "Smoothrunge" )

icounter <- 1
print(typeof(uniqFuncs))
print(uniqFuncs)
for (funcName in uniqFuncs){
  fileTarget <- str_c(as.character(funcName), ".tex")
  print(fileTarget)
  tikzDevice::tikz( file=all_of(fileTarget), standAlone=F, width=7, height=3)
  print("here")
  dev.off()
}



