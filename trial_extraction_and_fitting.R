##############################################################################################
#### EEG trial extraction and classification of .mat exported Event-Related EEG-data #########
##############################################################################################

#start clean
rm(list=ls())

#install and load packages
for (package in c("tidyverse", "R.matlab", "MASS", "eegkit", "signal", "keras", "tictoc")) {
  if (!require(package, character.only = TRUE, quietly = TRUE)) {
    install.packages(package)
    library(package, character.only = TRUE)
  }
}

#define working directory
setwd("~/p300Prediction/data/")
path <- getwd()

#scan in workingdir for input files
files <- dir()

extract_trials <- function(path, type = "train") {
  # initialize variables
  runs <- vector("list")
  
  srate = 2048                 # sampling rate of raw data in Hz
  reference <-  c(33,34)       # indices of channels used as reference
  filterorder <-  3   
  filtercutoff <- c(1/1024, 12/1024)  #high & lowpass
  #filt <-  butter(filterorder, W=filtercutoff, type=c("low","high"))
  filt <<- butter(filterorder, W=filtercutoff)
  freqz(filt)
  decimation <- 32           # downsampling factor
  n_samples  <<- srate/decimation #32           # number of (temporal) samples in a trial
  n_targets <- 0             # keeping track of number of target trials 
  n_nontargets <- 0          # keeping track of nontarget trials 
  
  # extract features from files in filelist
  p300_list_train <- vector("list")
  p300_list_test <- vector("list")
  non_targets_list <- vector("list")
  target_len <- c()
  non_target_len <- c()
  counter <- 0
  
for (i in 1:length(files)){
  cat("extrahiere Features aus EEG-Daten:", " File ", i, " von ", length(files), "\n") 
  # load data
  f <-  readMat(files[i])
  # rereference the data
  n_channels <-  dim(f$data)[1]
  ref <-  rep(mean(rowMeans(f$data[33:34,])), n_channels)
  f$data = f$data[1:34,] - rep(ref, dim(f$data)[2])
  
  # drop the mastoid channels
  f$data = f$data[1:32,]#1:32
  n_channels = dim(f$data)[1]
  
  # bandpass filter the data (with a forward-backward filter)
  for (j in 1:n_channels){
    f$data[j,] = filtfilt(filt$b,filt$a,f$data[j,])
  }
  
  # downsample the data (from 2048 Hz to 64 Hz)
  f$data <- f$data[,seq(1,dim(f$data)[2], decimation)] 
  
  # extract trials 
  # compute class labels
  n_trials = dim(f$events)[1]
  runs$x <- array(0, c(n_channels,n_samples,n_trials))
  #runs[i]$x <-  array(0, c(n_channels,n_samples,n_trials))
  events <- cbind(f$events)
  
  for (j in 1:n_trials){
    pos <-  round(difftime(ISOdatetime(f$events[j,1],f$events[j,2],f$events[j,3],f$events[j,4],f$events[j,5],f$events[j,6]),
                           ISOdatetime(f$events[1,1],f$events[1,2],f$events[1,3],f$events[1,4],f$events[1,5],f$events[1,6]))*(srate/decimation) + 1 + (0.4*srate/decimation)) 
    
    runs$x[,,j] <-  f$data[, pos:(pos + n_samples - 1)]
  }
  
  runs$y = array(0, n_trials);
  for (k in 1:n_trials){
    ifelse(f$stimuli[k] == f$target, runs$y[k] <- 1, runs$y[k] <- -1)
    runs$stimuli[k] <-  f$stimuli[k]
    runs$target[k]  <-  f$target
  }
  # update counters
  n_targets <-  n_targets + sum(runs$y == 1)
  n_nontargets <-  n_nontargets + sum(runs$y == -1)    
  cat("Targets: ", n_targets, "\n")
  cat("Non-Targets: ", n_nontargets, "\n")
  
  pos <- which(runs$y == 1)
  targets <- c()
  for (x in pos){
    targets <- rbind(targets, runs$x[32,,x]) #CH 32 = Cz #14???? runs$x[14,,x]
    target_len <- dim(targets)[1]
  }
  if (i > 1){
    counter <- counter +  target_len
  }
  non_target_pos <- which(runs$y == -1)
  non_targets <- c()
  for (x in non_target_pos){
    non_targets <- rbind(non_targets, runs$x[32,,x]) #CH 32 = Cz #14??runs$x[14,,x]
  }
  
  cat("targets \n")
  cat("counter: ", counter, "\n")
  for (k in 1:dim(targets)[1]) {
    if (type == "train") {
    p300_list_train[counter + k] <- list(targets[k,])
    } else {
    p300_list_test[counter + k] <- list(targets[k,])
    }
    non_targets_list[counter + k] <-list(non_targets[k,])
  }
  if (type =="train"){
    p300_list_train <<- p300_list_train 
    non_targets_list <<- non_targets_list
  } else if (type =="test") {
    p300_list_test <<- p300_list_test
    non_targets_list <<- non_targets_list
  }
  rm(targets)
  rm(non_targets)
  }
}

extract_trials(path, type = "train")

p300_list <- p300_list_train
p300_list <- Filter(Negate(is.null), p300_list)
targets <- length(p300_list)

p300_arr <- as.array(p300_list, c(n_samples,targets))
p300_df <- data.frame(matrix(unlist(p300_list), nrow = targets, byrow = TRUE), stringsAsFactors = FALSE)

p300_df_scaled <- scale(p300_df)

nontarget <- length(non_targets_list)
non_target_arr <- as.array(non_targets_list, c(n_samples, nontarget))
non_target_df <-  data.frame(matrix(unlist(non_targets_list), nrow = targets, byrow = T), stringsAsFactors = FALSE)
non_target_df_scaled <- scale(non_target_df)

#sample n Trials for the training dataframe
size <- 15
index <- sample(c(1:targets), size = size, replace = FALSE)

train_data <- c()
train_data <- rbind(p300_df_scaled[index, ], non_target_df_scaled[index, ])
train_data <- as.data.frame(train_data)
dim(train_data)

##preview data-quality
par(mfrow=c(5,6))
for (i in 1:dim(train_data)[1]){
  # p300 <- colMeans(train_data[i,])
  if (i <= dim(train_data)[1]/2){
  eegtime(seq(from = 10, 640, 10), train_data[i,,], plotzero = T, xlab = c("P300 Trial number: " , i), vcol = "limegreen")
  } else {
    eegtime(seq(from = 10, 640, 10), train_data[i,,], plotzero = T, xlab = c("Non-Traget Trial number: " , i), vcol = "red3")
  }
} 
dev.off()#train_data <- train_data[-c(2),]

#### bootstraped PCA: Principle Component Analysis
#### REMINDER: mit Reeler statt Adjungierter Matrix arbeiten (E statt V)
train_data_pca <- as.matrix(train_data)
svd <- fastSVD(train_data_pca, nv = 2*size) #single value decomposition
train_data_pca <- svd$u #rotate data


#train_data_pca <- t(svd$v) #rotate data
dim(train_data_pca)
# 
# par(mfrow=c(1,2))
# eegtime(seq(from=10, 640, 10), train_data[19,], plotzero = T, xlab=c("Number of trials", 1))
# eegtime(seq(from=10, 640, 10), train_data_pca[19,], plotzero = T, xlab=c("Number of trials", 1))

#look up training data
par(mfrow=c(3,5))
for (i in 1:15){
  # p300 <- colMeans(train_data[i,])
  eegtime(seq(from=10, 640, 10), train_data[i,], plotzero = T, xlab=c("Number of trials" , i))
} 
dev.off()#train_data <- train_data[-c(2),]

#n^trials p300
par(mfrow=c(5,2))
p300 <- colMeans(p300_df_scaled[1:40,])
eegtime(seq(10 ,640, 10),p300,plotzero = T)
dev.off()

#p300 over Trials
par(mfrow=c(1,5))
for (i in c(10,15,30,40)){
  p300 <- colMeans(p300_df_scaled[1:i,])
  eegtime(seq(10, 640, 10), p300, plotzero = T, xlab=c("Number of trials" , i))
} 

len <- dim(train_data_pca)[1]
#batch_size <- (n_samples)
batch_size <- dim(train_data_pca)[1]#24

#placeholder for tensorflow array
train_data <- array(0, dim=c(len, batch_size, 1))
#reshape train data to tensorflow array dims
train_data_pca <- array_reshape(as.matrix(train_data_pca), dim=c(len , batch_size, 1 ), "C" )
#saveRDS(train_data_pca,"~/p300Prediction/train_data.RDS")
#train_data_pca <- readRDS("~/p300Prediction/train_data.RDS")
dim(train_data_pca)

#create class labels for tensorflow
y <- c(rep(1, len/2), rep(0, len/2))
y.cat <- to_categorical(y, 2)
dim(y.cat)
#define tf model
model <- keras_model_sequential()
model %>% 
  layer_lstm(units = len , stateful = TRUE, batch_input_shape = c(1, batch_size, 1), return_sequences = FALSE) %>%
  layer_dense(units = dim(train_data_pca)[1], activation = "relu") %>%
  layer_dense(units = 2, activation = 'sigmoid')

model%>% 
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'rmsprop',
    metrics = c('accuracy')
  )

tic()
for (i in 1:200){
  model %>% fit(train_data_pca, y.cat, batch_size = 1, epochs = 1, shuffle = FALSE)
  model %>% reset_states()
  print(i)
}
toc()


#########################################################Validate Testdata######################################################################

#define working directory
setwd("~/p300Prediction/testdata/")
path <- "~/p300Prediction/testdata/"
files <- dir()

 #rm(list=c(targets, targets_test))
extract_trials(path, type="test")

p300_list_test <- Filter(Negate(is.null), p300_list_test)
targets_test <- length(p300_list_test)

p300_arr <- as.array(p300_list_test, c(n_samples,targets))
p300_df_test <- data.frame(matrix(unlist(p300_list_test), nrow = targets_test, byrow = TRUE), stringsAsFactors = FALSE)


p300_df_scaled <- scale(p300_df_test)

nontarget_test <- length(non_targets_list)
non_target_arr <- as.array(non_targets_list, c(n_samples, nontarget))
non_target_df <-  data.frame(matrix(unlist(non_targets_list), nrow=targets_test, byrow=T), stringsAsFactors=FALSE)
non_target_df_scaled <- scale(non_target_df)
#sample 15 Trials for the training dataframe
size <- targets_test
index <- sample(c(1:targets_test), size = size, replace = FALSE)

test_data <- c()
test_data <- rbind(p300_df_scaled[index,], non_target_df_scaled[index,])
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








