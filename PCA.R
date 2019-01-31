data <- read.csv("categorical_samples.csv")
data <- t(data)
data <- data.frame(data)
colnames(data) <- as.character(unlist(data[1,]))
head(colnames(data))
data <- data[-1, ]
features <-  data[ ,-c(1,2)]
head(features[1:5,1:5])
features_char <- lapply(features, as.character)
features_int <- lapply(features_char, as.numeric)
meta <- data.frame(features_int)
colnames(meta) <- colnames(features)
autoplot(prcomp(meta))
autoplot(prcomp(meta), data = data, colour = 'Multiclass')