# mixOmics

library(mixOmics)
library(gtools)

gut_16s_df <- read.table('./data/gut_16s_abundance.txt', header=T, sep='\t')
metabolome_df <- read.table('./data/metabolome_abundance.txt', header=T, sep='\t')
subjects <- read.csv('./data/subjects.csv', header=T, na.strings = "NA")

colnames(subjects)[1] <- 'SubjectID' 
colnames(subjects)

gut_16s_df['SubjectID'] <- substr(gut_16s_df[,1],1,7)
metabolome_df['SubjectID'] <- substr(metabolome_df[,1],1,7)

gut_16s_df <- merge(gut_16s_df,subjects,by.x='SubjectID', by.y ='SubjectID', all.x = TRUE)
metabolome_df <- merge(metabolome_df,subjects,by.x='SubjectID', by.y ='SubjectID', all.x = TRUE)

indexes <- intersect(gut_16s_df[,2], metabolome_df[,2])
indexes <- sort(indexes)
length(indexes)

gut_16s_df <- gut_16s_df[gut_16s_df[,2] %in% indexes,]
metabolome_df <- metabolome_df[metabolome_df[,2] %in% indexes,]
# sort by sampleID (index)
gut_16s_df <- gut_16s_df[order(gut_16s_df[,2]),]
metabolome_df <- metabolome_df[order(metabolome_df[,2]),]
dim(gut_16s_df); dim(metabolome_df)

X.g <- gut_16s_df[, -c(1:2,99:105)]
X.m <- metabolome_df[, -c(1:2,727:738)]
# Standardize
X.g <- scale(X.g)
X.m <- scale(X.m)
dim(X.g); dim(X.m)
rownames(X.g) <- rownames(X.m) <- indexes
indexes[1]

write.csv(gut_16s_df, './data/mixOmics_output/gut_16s_subj.csv', row.names = F)
write.csv(metabolome_df, './data/mixOmics_output/metabolome_subj.csv', row.names = F)

# Metabolites annotation
library(readxl)
m_annotation <- read_excel('./data/iPOP_Metablolite_Annotation.xlsx', 
                           sheet='Metablolite_annotation')
temp_merge <-merge(data.frame(m_name = colnames(X.m)),m_annotation,by.x='m_name',by.y='Compounds_ID',all.x=T,sort=F)
colnames(X.m) <- temp_merge$Metabolite

# Training, testing split

train_val_test_indexes <- function(covariate) {
  index_folder <- paste0('./data/index/', covariate)
  indexes_train <- as.vector(read.table(paste0(index_folder, '/indexes_train.txt'))[,1])
  indexes_val <- as.vector(read.table(paste0(index_folder, '/indexes_val.txt'))[,1])
  indexes_test <- as.vector(read.table(paste0(index_folder, '/indexes_test.txt'))[,1])
  return(list(indexes_train,  indexes_val, indexes_test))
}
indexes_iris <- train_val_test_indexes('IR_IS_classification')
indexes_train <- indexes_iris[[1]]
indexes_val <- indexes_iris[[2]]
indexes_test <- indexes_iris[[3]]

train_val_split <- function(data, indexes_train, indexes_val, indexes_test) {
  data.train <- data[rownames(data) %in% indexes_train, ]
  data.val <- data[rownames(data) %in% indexes_val, ]
  data.test <- data[rownames(data) %in% indexes_test, ]
  return(list(data.train, data.val, data.test))
}

#####################################################################################################################################
# 3 methods from mixOmics: sPLSDA (separate for microbiome and metabolome), DIABLO, sPLS (X:microbiome, Y:metabolome).
#####################################################################################################################################

# 1. sPLSDA prediction (separate for microbiome and metabolome)

predict_splsda <- function(data_train, data_val, data_test, data_covariate, covariate, indexes_train, indexes_val, indexes_test) {
  label_train <- as.vector(data_covariate[data_covariate[,2] %in% indexes_train,][,covariate])
  label_val <- as.vector(data_covariate[data_covariate[,2] %in% indexes_val,][,covariate])
  label_test <- as.vector(data_covariate[data_covariate[,2] %in% indexes_test,][,covariate])
  splsda <- splsda(data_train, label_train)
  prediction_val <- predict(splsda, data_val)
  prediction_test <- predict(splsda, data_test)
  return(list(splsda, prediction_val, prediction_test, label_train, label_val, label_test))
}

cal_accuracy <- function(data_train, data_val, data_test, data_covariate, covariate, indexes_train, indexes_val, indexes_test) {
  splsda_X <- predict_splsda(data_train, data_val, data_test, data_covariate, covariate, indexes_train, indexes_val, indexes_test)
  predict_X_val <- splsda_X[[2]]
  predict_X_test <- splsda_X[[3]]
  class_pred_X_val <- predict_X_val$class$max.dist
  class_pred_X_test <- predict_X_test$class$max.dist
  accuracy_X_val <- mean(class_pred_X_val[,1] == splsda_X[[5]], na.rm = T)
  accuracy_X_test <- mean(class_pred_X_test[,1] == splsda_X[[6]], na.rm = T)
  return(list(accuracy_X_val, accuracy_X_test, splsda_X))
}

covariate_list <- c('IR_IS_classification', 'Sex', 'Race')
accuracy_result_val <- accuracy_result_test <- matrix(0, nrow=length(covariate_list), ncol=2)
for (i in 1:length(covariate_list)) {
  indexes_train_val_test <- train_val_test_indexes(covariate_list[i])
  indexes_train <- indexes_train_val_test[[1]]
  indexes_val <- indexes_train_val_test[[2]]
  indexes_test <- indexes_train_val_test[[3]]
  X.g.split <- train_val_split(X.g, indexes_train, indexes_val, indexes_test)
  X.m.split <- train_val_split(X.m, indexes_train, indexes_val, indexes_test)
  X.g.train <- X.g.split[[1]]; X.g.val <- X.g.split[[2]]; X.g.test <- X.g.split[[3]]
  X.m.train <- X.m.split[[1]]; X.m.val <- X.m.split[[2]]; X.m.test <- X.m.split[[3]]
  
  accuracy_result_g <- cal_accuracy(X.g.train, X.g.val, X.g.test, gut_16s_df, covariate_list[i], indexes_train, indexes_val, indexes_test)
  accuracy_result_m <- cal_accuracy(X.m.train, X.m.val, X.m.test, metabolome_df, covariate_list[i], indexes_train, indexes_val, indexes_test)
  accuracy_result_val[i,1] <- accuracy_result_g[[1]]; accuracy_result_val[i,2] <- accuracy_result_m[[1]]
  accuracy_result_test[i,1] <- accuracy_result_g[[2]]; accuracy_result_test[i,2] <- accuracy_result_m[[2]]
  
  splsda_X_g <- accuracy_result_g[[3]]
  splsda_X_m <- accuracy_result_m[[3]]
  write.csv(splsda_X_g[[1]]$variates$X, paste0('./data/mixOmics_output/splsda_g_train_', covariate_list[i],'.csv'), 
            row.names=indexes_train)
  write.csv(splsda_X_m[[1]]$variates$X, paste0('./data/mixOmics_output/splsda_m_train_', covariate_list[i],'.csv'), 
            row.names=indexes_train)
  write.csv(splsda_X_g[[2]]$variates, paste0('./data/mixOmics_output/splsda_g_val_', covariate_list[i],'.csv'), 
            row.names=indexes_val)
  write.csv(splsda_X_m[[2]]$variates, paste0('./data/mixOmics_output/splsda_m_val_', covariate_list[i],'.csv'), 
            row.names=indexes_val)
  write.csv(splsda_X_g[[3]]$variates, paste0('./data/mixOmics_output/splsda_g_test_', covariate_list[i],'.csv'), 
            row.names=indexes_test)
  write.csv(splsda_X_m[[3]]$variates, paste0('./data/mixOmics_output/splsda_m_test_', covariate_list[i],'.csv'), 
            row.names=indexes_test)
}
accuracy_result_val <- data.frame(accuracy_result_val)
dimnames(accuracy_result_val) <- list(covariate_list, c('gut_16s', 'metabolome'))
cat('Validation:\n')
print(100*round(accuracy_result_val,4))

accuracy_result_test <- data.frame(accuracy_result_test)
dimnames(accuracy_result_test) <- list(covariate_list, c('gut_16s', 'metabolome'))
cat('Testing:\n')
print(100*round(accuracy_result_test,4))

###########################################################
# 2. DIABLO
data.list <- list(gut_16s = X.g, metabolome=X.m)
### Default design matrix: fully connected
design.mat <- matrix(1, ncol = length(data.list), nrow = length(data.list),
                     dimnames = list(names(data.list), names(data.list)))
diag(design.mat) <- 0
design.mat

diablo_model <- function(datalist_train, datalist_val, datalist_test, data_covariate, 
                         covariate, indexes_train, indexes_val, indexes_test, design.mat=design.mat){
  label_train <- as.vector(data_covariate[data_covariate[,2] %in% indexes_train,][,covariate])
  label_val <- as.vector(data_covariate[data_covariate[,2] %in% indexes_val,][,covariate])
  label_test <- as.vector(data_covariate[data_covariate[,2] %in% indexes_test,][,covariate])
  diablo <- block.splsda(datalist_train, label_train, design=design.mat)
  prediction_val <- predict(diablo, datalist_val)
  prediction_test <- predict(diablo, datalist_test)
  return(list(diablo, prediction_val, prediction_test, label_train, label_val, label_test))
}

cal_accuracy_diablo <- function(datalist_train, datalist_val, datalist_test, data_covariate, 
                                covariate, indexes_train, indexes_val, indexes_test, design.mat=design.mat){
  diablo_predict <- diablo_model(datalist_train, datalist_val, datalist_test, data_covariate, 
                                 covariate, indexes_train, indexes_val, indexes_test, design.mat=design.mat)
  predict_val <- diablo_predict[[2]]
  predict_test <- diablo_predict[[3]]
  class_pred_val <- predict_val$class$max.dist
  class_pred_test <- predict_test$class$max.dist
  accuracy1_val <- mean(class_pred_val[[1]][,1] == diablo_predict[[5]], na.rm = T)
  accuracy2_val <- mean(class_pred_val[[2]][,1] == diablo_predict[[5]], na.rm = T)
  accuracy1_test <- mean(class_pred_test[[1]][,1] == diablo_predict[[6]], na.rm = T)
  accuracy2_test <- mean(class_pred_test[[2]][,1] == diablo_predict[[6]], na.rm = T)
  return(list(c(accuracy1_val, accuracy2_val), c(accuracy1_test, accuracy2_test), diablo_predict))
}

covariate_list <- c('IR_IS_classification', 'Sex', 'Race')
accuracy_result_diablo_val <- accuracy_result_diablo_test <- matrix(0, nrow=length(covariate_list), ncol=2)
for (i in 1:length(covariate_list)) {
  indexes_train_val_test <- train_val_test_indexes(covariate_list[i])
  indexes_train <- indexes_train_val_test[[1]]
  indexes_val <- indexes_train_val_test[[2]]
  indexes_test <- indexes_train_val_test[[3]]
  X.g.split <- train_val_split(X.g, indexes_train, indexes_val, indexes_test)
  X.m.split <- train_val_split(X.m, indexes_train, indexes_val, indexes_test)
  X.g.train <- X.g.split[[1]]; X.g.val <- X.g.split[[2]]; X.g.test <- X.g.split[[3]]
  X.m.train <- X.m.split[[1]]; X.m.val <- X.m.split[[2]]; X.m.test <- X.m.split[[3]]
  datalist_train <- list(gut_16s = X.g.train, metabolome = X.m.train)
  datalist_val <- list(gut_16s = X.g.val, metabolome = X.m.val)
  datalist_test <- list(gut_16s = X.g.test, metabolome = X.m.test)
  
  accuracy_result_diablo <- cal_accuracy_diablo(datalist_train, datalist_val, datalist_test, gut_16s_df, 
                                                    covariate_list[i], indexes_train, indexes_val, indexes_test)
  accuracy_result_diablo_val[i,] <- accuracy_result_diablo[[1]]
  accuracy_result_diablo_test[i,] <- accuracy_result_diablo[[2]]
  
  diablo_predict <- accuracy_result_diablo[[3]]
  write.csv(diablo_predict[[1]]$variates$gut_16s, paste0('./data/mixOmics_output/diablo_g_train_', covariate_list[i],'.csv'), 
            row.names=indexes_train)
  write.csv(diablo_predict[[1]]$variates$metabolome, paste0('./data/mixOmics_output/diablo_m_train_', covariate_list[i],'.csv'), 
            row.names=indexes_train)
  write.csv(diablo_predict[[2]]$variates$gut_16s, paste0('./data/mixOmics_output/diablo_g_val_', covariate_list[i],'.csv'), 
            row.names=indexes_val)
  write.csv(diablo_predict[[2]]$variates$metabolome, paste0('./data/mixOmics_output/diablo_m_val_', covariate_list[i],'.csv'), 
            row.names=indexes_val)
  write.csv(diablo_predict[[3]]$variates$gut_16s, paste0('./data/mixOmics_output/diablo_g_test_', covariate_list[i],'.csv'), 
            row.names=indexes_test)
  write.csv(diablo_predict[[3]]$variates$metabolome, paste0('./data/mixOmics_output/diablo_m_test_', covariate_list[i],'.csv'), 
            row.names=indexes_test)
}
accuracy_result_diablo_val <- data.frame(accuracy_result_diablo_val)
dimnames(accuracy_result_diablo_val) <- list(covariate_list, c('gut_16s', 'metabolome'))
cat('Validation:\n')
print(100*round(accuracy_result_diablo_val,4))

accuracy_result_diablo_test <- data.frame(accuracy_result_diablo_test)
dimnames(accuracy_result_diablo_test) <- list(covariate_list, c('gut_16s', 'metabolome'))
cat('Testing:\n')
print(100*round(accuracy_result_diablo_test,4))

###########################################################
# 3. sPLS (X:microbiome, Y:metabolome)

# Use all 720 samples
gm_spls <- spls(X.g, X.m, ncomp=10)

write.csv(gm_spls$variates$X, './data/mixOmics_output/spls_g.csv', row.names=indexes)
write.csv(gm_spls$variates$Y, './data/mixOmics_output/spls_m.csv', row.names=indexes)

gm_spls$variates$Y
X.m %*% gm_spls$loadings$Y
gm_spls$variates$Y %*% t(gm_spls$loadings$Y)

X.m

