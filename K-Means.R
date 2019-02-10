data <- read.table(file = "early_cancer.csv", row.names = 1, sep = ",")
colnames(data) <- as.character(unlist(data[1,]))
head(colnames(data))
data <- data[-1, ]
features <-  data[ ,-c(1,2)]
head(features[1:5,1:5])
k2 <- kmeans(features, centers = 2, nstart = 25)
str(k2)
fviz_cluster(k2, data = features)

