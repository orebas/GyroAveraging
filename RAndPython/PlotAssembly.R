#detach(package:plyr)   
library(dplyr)
library(ggplot2)
library(tidyverse)
#library(plyr)
library(readr)
library(tikzDevice)
library(ggrepel)
library(directlabels)
#detach(package:plyr)   

data_all <- list.files(path = ".",     # Identify all csv files in folder
                       pattern = "temp.csv", full.names = TRUE) %>% 
  lapply(read_csv) %>%                                            # Store all files in list
  bind_rows     


print(data_all)

uniqModels <- unique(data_all[,2])
uniqFuncs <- unlist(as.list(unique(data_all[,1])))

data_all = filter (data_all, calculator != "CPUDCTnopad")
data_all = filter (data_all, N != 8)
data_double = filter(data_all, bytes == 8)
data_float = filter(data_all, bytes == 4)


HzOnly <- data_all %>% 
  filter (calculator != "CPUDCTnopad") %>% 
  filter (calculator != "CPUBicubicQuad")  %>% 
  filter (calculator != "CPULinearQuad") %>%
  group_by (N, calculator, bytes) %>% 
  summarize(speed = median(calcHz,na.rm=TRUE))
HzOnlyWide <- pivot_wider(HzOnly, names_from = bytes, values_from=speed)
HzOnlyWide <- mutate(HzOnlyWide, ratio = `4` / `8`)

HzOnlyWide2 = HzOnly %>%
  filter(bytes == 8) %>%
  filter(N<290) %>%
  pivot_wider(names_from=calculator, values_from=speed)
RatioChart <- 
  HzOnlyWide2 %>%
  mutate(LinearSpeedup = GPULinearSparse/CPULinearSparse) %>%
  mutate(BicubicSpeedup = GPUBicubicSparse/CPUBicubicSparse) %>%
  mutate(ChebSpeedup = GPUChebDense/CPUChebDense) %>%
  select(LinearSpeedup,BicubicSpeedup,ChebSpeedup) %>%
  pivot_longer(!N,names_to="calculator", values_to="Speedup")
print(RatioChart)

tikzDevice::tikz( file="DoubleFloatRatio.tex", 
                  standAlone=F, 
                  width=9, 
                  height=6.5)
myPlot <- ggplot(data=HzOnlyWide,
                 aes( N,ratio, color=calculator,linetype=calculator))+
  geom_line()+geom_point() +
  scale_y_continuous(trans="log10") +#, breaks = c(1.0,.1,.01,.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11)) +
  #scale_x_continuous(trans="log2")+#,breaks=c(2,4,8 ,16,32,64,128,256,512,1024)) +
  ggtitle(funcName)+geom_dl(aes(label=calculator), method=list("maxvar.points", "bumpup",cex=0.8))+
  scale_colour_discrete(guide="none")+
  scale_linetype_discrete(guide="none")+
  coord_cartesian(clip="off")
print (myPlot)
dev.off()

tikzDevice::tikz( file="GPUImpact.tex", 
                  standAlone=F, 
                  width=9, 
                  height=6.5)
myPlot <- ggplot(data=RatioChart,
                 aes( N,Speedup, color=calculator,linetype=calculator))+
  geom_line()+geom_point() +
  #scale_y_continuous(trans="log10") +#, breaks = c(1.0,.1,.01,.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11)) +
  #scale_x_continuous(trans="log2")+#,breaks=c(2,4,8 ,16,32,64,128,256,512,1024)) +
  ggtitle(funcName)+geom_dl(aes(label=calculator), method=list("maxvar.points", "bumpup",cex=0.8))+
  scale_colour_discrete(guide="none")+
  scale_linetype_discrete(guide="none")+
  coord_cartesian(clip="off")+
  facet_wrap( . ~ calculator, scales="free", ncol=1)
print (myPlot)
dev.off()






print(typeof(uniqFuncs))
print(uniqFuncs)
for (funcName in uniqFuncs){
  fileTarget <- paste0({{funcName}}, ".tex")
  print(fileTarget)
  tikzDevice::tikz( file=all_of(fileTarget), 
                    standAlone=F, 
                    width=9, 
                    height=6.5)
  filteredData = data_double[data_double$functionName == funcName,]
  myPlot <- ggplot(data=filteredData,
                   aes( N,maxError, color=calculator,linetype=calculator))+
    geom_line()+geom_point() +
    scale_y_continuous(trans="log10") +#, breaks = c(1.0,.1,.01,.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11)) +
    scale_x_continuous(trans="log2")+#,breaks=c(2,4,8 ,16,32,64,128,256,512,1024)) +
    ggtitle(funcName)+geom_dl(aes(label=calculator), method=list("maxvar.points", "bumpup",cex=0.8))+
    scale_colour_discrete(guide="none")+
    scale_linetype_discrete(guide="none")+
    coord_cartesian(clip="off")
  print (myPlot)
  dev.off()
  
  fileTarget <- paste0({{funcName}}, "S.tex")
  print(fileTarget)
  tikzDevice::tikz( file=all_of(fileTarget), 
                    standAlone=F, 
                    width=9, 
                    height=6.5)
  filteredData = data_double[data_double$functionName == funcName,]
  myPlot2 <- ggplot(data=filteredData,
                   aes( calcHz,maxError, color=calculator,linetype=calculator))+
    geom_line()+geom_point() +
    scale_y_continuous(trans="log10")+#, breaks = c(1.0,.1,.01,.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11)) +
    scale_x_continuous(trans="log10")+#,breaks=c(.1,1,10,100,1000,10000,100000,1000000)) +
    ggtitle(funcName)+geom_dl(aes(label=calculator), method=list("maxvar.points", "bumpup",cex=0.8))+
    scale_colour_discrete(guide="none")+
    scale_linetype_discrete(guide="none")+
    coord_cartesian(clip="off")
  print (myPlot2)
  dev.off()

  
}



