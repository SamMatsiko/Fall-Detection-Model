#Installing and loading the required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")

# if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(755, sample.kind="Rounding")

#Reading the dataset from github
urlfile="https://raw.githubusercontent.com/SamMatsiko/Fall-Detection-Model/master/falldeteciton.csv
"

FallData<-read_csv(url(urlfile))


#Changing the target label variable to factor class
FallData$ACTIVITY<-as.factor(FallData$ACTIVITY)

#Scaling predictors
FallData[,2:7]<-scale(FallData[,2:7])


#Partioning away the validation set that will be used to test after all the modelling.  
Partition_set <- createDataPartition(y = FallData$ACTIVITY , times = 1,p = 0.2, list = FALSE)
main_set <- FallData[-Partition_set,]
validation_set <- FallData[Partition_set,]

#Subpartitioning the remaining set into train and test sets 
test_index <- createDataPartition(y = main_set$ACTIVITY, times = 1,p = 0.5, list = FALSE)
train_set <- main_set[-test_index,]
test_set <- main_set[test_index,]

#Training with svm on train_Set and tests on test_set
svm_fit <- svm(ACTIVITY ~ . ,train_set)
y_hat_svm <- predict(svm_fit, test_set)
cm_svm<-confusionMatrix(y_hat_svm,test_set$ACTIVITY)$overall["Accuracy"]

#Creating a table that will store accuracy results of each ML algorithm
Accuracy_results <- tibble(method = "Training with Svm",  Accuracy= cm_svm)
Accuracy_results


#Training with knn on train_Set and tests on test_set
knn_fit<-train(ACTIVITY~.,data = train_set,method="knn")
y_hat_knn<-predict(knn_fit,test_set)
cm_knn<-confusionMatrix(y_hat_knn,test_set$ACTIVITY)$overall["Accuracy"]
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(method="Training with knn ",
                                     Accuracy = cm_knn))
Accuracy_results

#Training with random forest on train_Set and tests on test_set

rf_fit <- randomForest(ACTIVITY~., data = train_set)

y_hat_rf<-predict(rf_fit,test_set)

cm_rf<-confusionMatrix(y_hat_rf,test_set$ACTIVITY)$overall["Accuracy"]
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(method="Training with random forest",
                                     Accuracy = cm_rf))
Accuracy_results


#Final test on validation set using random forest

rf_fit_main <- randomForest(ACTIVITY~ ., data = main_set)

y_hat_rf_main<-predict(rf_fit_main,validation_set)

cm_rf_main<-confusionMatrix(y_hat_rf_main,validation_set$ACTIVITY)$overall["Accuracy"]
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(method="Final test on Random forest with validation set",
                                     Accuracy = cm_rf_main))
Accuracy_results












