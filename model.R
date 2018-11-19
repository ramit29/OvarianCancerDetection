setwd("~/Documents/OvarianCancerDetection/")
library(sigFeature)
data <- read.csv("data.csv")
y <- data[1:95,1]
x <- data[1:95,2:256]
system.time(sigfeatureRankedList <- sigFeature(x, y))
library(e1071)
selectedfeature <- sigfeatureRankedList[1:25]
sigFeature.model=svm(x[ ,selectedfeature], y,type="C-classification", kernel="linear")
pred <- predict(sigFeature.model, x[ ,selectedfeature])
table(pred,y)
