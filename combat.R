setwd('~/Documents/OvarianCancerDetection/summer/OneDrive_1_5-27-2019/temp/final_files/')
data <- read.csv('combatbatch_neg.csv')
features <-  data[ ,-c(1,2,3,4)]
feat_t <- t(features)
library(sva)
Combat(dat=feat_t,batch=data$Source,mod=data$Class)
combat_loc <- ComBat(dat=feat_t,batch=data$Location,mod=data$Class)
combat_loc <- t(combat_loc)
write.csv(combat_loc,'neg_locnorm.csv')
