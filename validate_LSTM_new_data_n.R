###### LSTM Validation

##define working directory
setwd("~/p300Prediction/testdata/")
path <- "~/p300Prediction/testdata/"
files <- dir()

#rm(list=c(targets, targets_test))
extract_trials(path, type="test", channel = 32)

p300_list_test <- Filter(Negate(is.null), p300_list_test)
targets_test <- length(p300_list_test)

p300_arr <- as.array(p300_list_test, c(n_samples,targets))
p300_df_test <- data.frame(matrix(unlist(p300_list_test), nrow = targets_test, byrow = TRUE), stringsAsFactors = FALSE)


p300_df_scaled <- scale(p300_df_test)

nontarget_test <- length(non_targets_list)
non_target_arr <- as.array(non_targets_list, c(n_samples, nontarget))
non_target_df <- data.frame(matrix(unlist(non_targets_list), nrow = targets_test, byrow = TRUE), stringsAsFactors=FALSE)
non_target_df_scaled <- scale(non_target_df)
#sample 15 Trials for the training dataframe
size <- targets_test
index <- sample(c(1:targets_test), size = size, replace = FALSE)

test_data <- c()
test_data <- rbind(p300_df_scaled[index, ], non_target_df_scaled[index, ])
dim(test_data)
####PCA
test_data_pca <- as.matrix(test_data)
test_svd <- fastSVD(test_data_pca, nv=30)
#test_data_pca <- t(test_svd$v) #sample PCs for a wide matrix are the right singular vectors
test_data_pca <- test_svd$u
test <- array_reshape(as.matrix(test_data_pca), dim=c(dim(test_data_pca)[1] , batch_size, 1 ), "C" )
#saveRDS(test,"testdata.RDS")
#test <- readRDS("~/p300Prediction/testdata.RDS")
predicted <- keras::predict_classes(model, test, batch_size = 1)
observed <- c(rep(1, dim(test_data_pca)[1]/2), rep(0, dim(test_data_pca)[1]/2))
length(observed)

cat("accuracy: ", sum(predicted == observed)/length(predicted)*100, "%")

