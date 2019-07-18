log_transformation <- function(file){
  data <- read.csv(file)
  features <- data[,-c(1,2,3,4)]
  meta_data <- data[,c(1,2,3,4)]
  rownames(metadata) <- rownames(data)
  log2 <- log(as.matrix(features),2)
  final_data <- bind_cols(meta_data,as.data.frame(log2))
  return(final_data)
}