library(caret)
library(e1071) #SVM
library(ggplot2)
library(rJava)
library(RWeka)
library(ROCR)

#PLEASE SET THE PATH OF THE DATA FILE TO APPROPRIATE FOLDER!!!!

Data_csv <- read.csv("~/IdmProject1/R/Data_original.csv", colClasses=c(NA,"NULL",NA,NA,NA,NA), header = TRUE)
#Training and test set creation

divideDataNew<-function(x)
{
  #PLEASE SET THE PATH OF THE DATA FILE TO APPROPRIATE FOLDER
  # Data_csv <- read.csv("~/IdmProject1/R/Data_original.csv", colClasses=c(NA,"NULL",NA,NA,NA,NA), header = TRUE)
  set.seed(x)
  intrain <- createDataPartition(y=Data_csv$Continent, p= 0.8, list = FALSE)
  training <- Data_csv[intrain,]
  testing <- Data_csv[-intrain,]
  dim(training); dim(testing);
  #check whether any NA value exists or not
  anyNA(Data_csv)
  return(list(training=training,testing=testing))
}

#Eval measures
# This function computes recall for multi-class learning problem
recall_multi<-function(M) {
  k = dim(M)[1]
  rk = rep(0, k)
  n_total = sum(M)
  for (i in 1:k) {
    M2 = matrix(0, 2, 2)
    M2[1,1] = M[i,i]
    M2[1,2] = sum(M[i,]) - M[i,i]
    M2[2,1] = sum(M[,i]) - M[i,i]
    M2[2,2] = n_total - M2[1,1] - M2[1,2] - M[2,1]
    
    r2 = recall_binary(M2)
    rk[i] = r2
  }
  return(rk)
}


recall_binary<-function(M) {
  if (sum(M[1,])!=0) {
    r = M[1,1] / sum(M[1,])
  }else {
    r = 0
  }
  
  return(r)
}

# This function computes precision for multi-class learning problem

precision_multi<-function(M) {
  k = dim(M)[1]
  pk = rep(0, k)
  n_total = sum(M)
  for (i in 1:k) {
    M2 = matrix(0, 2, 2)
    M2[1,1] = M[i,i]
    M2[1,2] = sum(M[i,]) - M[i,i]
    M2[2,1] = sum(M[,i]) - M[i,i]
    M2[2,2] = n_total - M2[1,1] - M2[1,2] - M[2,1]
    
    p2 = precision_binary(M2)
    pk[i] = p2
  }
  return(pk)
}


# computes the precision for binary classification
precision_binary<-function(M) {
  if (sum(M[,1])!=0) {
    p = M[1,1] / sum(M[,1])
  }else {
    p = 0
  }
  
  return(p)
}

# This function computes F1 for multi-class learning problem

F1_multi<-function(M) {
  k = dim(M)[1]
  fk = rep(0, k)
  n_total = sum(M)
  for (i in 1:k) {
    M2 = matrix(0, 2, 2)
    M2[1,1] = M[i,i]
    M2[1,2] = sum(M[i,]) - M[i,i]
    M2[2,1] = sum(M[,i]) - M[i,i]
    M2[2,2] = n_total - M2[1,1] - M2[1,2] - M[2,1]
    
    f1 = F1_binary(M2)
    fk[i] = f1
  }
  return(fk)
}

# This function computes the F1 for binary classification,

F1_binary<-function(M) {
  p = precision_binary(M)
  r = recall_binary(M)
  if (p==0&&r==0) {
    f = 0
  }else {
    f = 2 * p * r / (p+r)
  }
  
  return(f)
}

# This function computes macro-averaged F1 for multi-class learning problem

macroF1 <-function(M) {
  fk = F1_multi(M)
  return(mean(fk))
}


# This function computes micro-averaged F1 for multi-class learning problem

microF1 <-function(M) {
  k = dim(M)[1]
  M2_sum = matrix(0, 2, 2)
  
  n_total = sum(M)
  for (i in 1:k) {
    M2 = matrix(0, 2, 2)
    M2[1,1] = M[i,i]
    M2[1,2] = sum(M[i,]) - M[i,i]
    M2[2,1] = sum(M[,i]) - M[i,i]
    M2[2,2] = n_total - M2[1,1] - M2[1,2] - M[2,1]
    
    M2_sum = M2_sum + M2
  }
  F1 = F1_binary(M2_sum)
  return(F1)
}


#TRAINING CLASSIFIERS

#K Nearest Neighbors(KNN)
#confMatrix<-table()
learnKnn <- function(training)
{
  #training<-divideData()$training1
  training[["Continent"]] = factor(training[["Continent"]]) #conversion of V1 integer variable to factor variable


  #Training & Preprocessing.We just need to pass different parameter values for different algorithms.
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3) #It controls the computational nuances of the train() method

  set.seed(3333)

  knn_fit <- train(Continent ~., data = training, method = "knn",
                   trControl=trctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)

  print(knn_fit) #knn classifier
  return(knn_fit)

}

#SVM

#SVM Radial using e0107
learnSVMradial <- function(training)
{
  # training<-divideData()$training2
  # testing<-divideData()$testing2

  #Tuning
  set.seed(123)

  tmodel<-tune(svm,Continent~.,data=training,ranges=list(epsilon=seq(0,1,0.01),cost=2^(2:7)))
  summary(tmodel)#best cost and gamma values
  plot(tmodel)

  mymodel1 <-tmodel$best.model
  return(mymodel1)

}

#C4.5 Decision Tree --Check training and test sets
decisionTreeC45<-function(training)
{
  # training<-divideData()$training5
  # testing<-divideData()$testing5
  #set.seed(123)

  tree_j48 <- J48(Continent ~ ., data = training,control = Weka_control(),options=parse_Weka_digraph)#J48 generates unpruned or pruned C4.5 decision trees (Quinlan, 1993).
 # plot(tree_j48)
  #print(tree_j48)
  return(tree_j48)
}

JripDecisionTreeNew<-function(training)
{
  set.seed(2192)
  # training<-divideData()$training5
  # testing<-divideData()$testing5
  tree_rip <- JRip(Continent ~ ., data = training ,control = Weka_control(O=50,F=10))#J48 generates unpruned or pruned C4.5 decision trees (Quinlan, 1993).
 ## #plot(tree_rip)
  print(tree_rip)
  return(tree_rip)

}

#Predict

classiPredictKNN<- function(classi_fit,testing)
{
  
  test_pred <- predict(classi_fit, newdata = testing)
  #print("test pred here")
  #print(test_pred)
   plot(test_pred)

  #Test set Statistics
  cl<-factor(testing$Continent)
  #print(cl)
  tab<-table(test_pred,cl)
  tabcl<-c("Africa","Asia","Australia","Europe","North America","South America")

  #Accuracy
  accuracy=sum(diag(tab))/sum(tab)
  print(accuracy)

  #recall
  recall_M=recall_multi(tab)
  
  #precision
  precision=precision_multi(tab)
  
  #F1 Measure
  f1=F1_multi(tab)
 
  #Macro averaged
  microF1=microF1(tab)
  
  #Test set Statistics
  conf=confusionMatrix(test_pred, testing$Continent )
  print(conf)  #check accuracy
  return(list(acc=accuracy,f1Measure=f1,recall=recall_M,prec=precision,microf1=microF1))

}


#predict classes for test set using knn classifier
classiPredict<- function(classi_fit,testing)
{
  test_pred <- predict(classi_fit, newdata = testing)
  #plot(test_pred)

  #Test set Statistics
  cl<-factor(testing$Continent)
  #print(cl)
  tab<-table(test_pred,cl)
  tabcl<-c("Africa","Asia","Australia","Europe","North America","South America")
  #print(tab)
  #precision=(diag(tab)/rowsum(tab,tabcl))*100

  #Accuracy
  accuracy=sum(diag(tab))/sum(tab)
  print(accuracy)

  #recall
  recall_M=recall_multi(tab)
  # print("Recall(Positive Prediction Value)")
  # print(recall_M)
  #plot(recall_M,xlab="Continents",ylab="Recall Measure")

  #precision
  precision=precision_multi(tab)
  # print("Precision Value(Sensitivity)")
  # print(precision)
  # plot(precision,xlab="Continents",ylab="Precision Measure")

  #F1 Measure
  f1=F1_multi(tab)
  # print("F1 Measure")
  # print(f1)
  # plot(f1,xlab="Continents",ylab="F1 Measure")

  #Macro averaged
  microF1=microF1(tab)
  # print("microF1 Average ")
  # print(microF1)

  #Test set Statistics
  conf=confusionMatrix(test_pred, testing$Continent )
  #write(conf,file=~/IdmProject1/R/ConfusionMatrix.csv)  #check accuracy
  print(conf)
  cat("\n")

  return(list(acc=accuracy,f1Measure=f1,recall=recall_M,prec=precision,microf1=microF1))

}

#GLOBAL
#KNN CLASSIFIER RESULTS

knnAccuracy<-list()
KnnResults<-data.frame()
new_row<-list()

svmAccuracy<-list()
svmResults<-data.frame()
svmnew_row<-list()

ripperAccuracy<-list()
ripperResults<-data.frame()
ripper_new_row<-list()

c45Accuracy<-list()
c45Results<-data.frame()
c45_new_row<-list()


continentList=data.frame(continent=rep(c("Africa","Asia","Australia","Europe","North America","South America"),6))
for(x in c(0:5))
{
  set.seed(x+200)
  training <-divideDataNew(x)$training
  testing<-divideDataNew(x)$testing
  
  #KNN
  print(paste0(" KNN Model",x))
  knn_fit <- learnKnn(training)
  plot(knn_fit)
  print(paste0(" KNN Model",x))
  knnAccuracy[x]<-classiPredictKNN(knn_fit,testing)$acc
  dat<-data.frame(classiPredictKNN(knn_fit,testing))
  dat$x<-x

  new_row[[x+1]]<-data.frame(dat)
 

#RIPPER CLASSIFICATION

  print(paste0("Ripper Model",x))
  ripper_fit <- JripDecisionTreeNew(training)
  #plot(ripper_fit)
  print(paste0("Ripper Model",x))
  ripperAccuracy[x]<-classiPredict(ripper_fit,testing)$acc
  dat1<-data.frame(classiPredict(ripper_fit,testing))
  dat1$x<-x

  ripper_new_row[[x+1]]<-data.frame(dat1)


#C4.5 CLASSIFICATION MODEL


  print(paste0("C4.5 Model",x))
  c45_fit <- decisionTreeC45(training)
  plot(c45_fit)
  print(paste0("C4.5 Model",x))
  c45Accuracy[x]<-classiPredict(c45_fit,testing)$acc
  dat2<-data.frame(classiPredict(c45_fit,testing))
  dat2$x<-x

  c45_new_row[[x+1]]<-data.frame(dat2)


#SVM CLASSIFIER

  print(paste0("SVM Model",x))
  svm_fit <- learnSVMradial(training)
 # Classification Plot
  plot(svm_fit,data=training,Male.life.expectancy.at.birth~Female.life.expectancy.at.birth, slice = list(Overall.life.expectancy.at.birth=60))

  print(paste0("SVM Model",x))
  svmAccuracy[x]<-classiPredict(svm_fit,testing)$acc
  dat3<-data.frame(classiPredict(svm_fit,testing))
  dat3$x<-x

  svmnew_row[[x+1]]<-data.frame(dat3)
}
  
  KnnResults<-do.call(rbind,new_row)
  KnnResults$Continent=continentList
  
  ripperResults<-do.call(rbind,ripper_new_row)
  ripperResults$Continent=continentList
  
  c45Results<-do.call(rbind,c45_new_row)
  c45Results$Continent=continentList
  
  svmResults<-do.call(rbind,svmnew_row)
  svmResults$Continent=continentList



#RESULTS and ACCURACIES
#KNN
  cat("\n")
print("Accuracy Measures for classifiers:")
#print(table(knnAccuracy))
cat("\n")
#Accuracy Measures acc=accuracy,f1=f1,recall=recall_M,prec=precision,microf1=microF1
print("Best Accuracy obtained using KNN :")
bestAccuracyKnn<-max(unlist(knnAccuracy))
print(bestAccuracyKnn)
meanAccuracyKNN<-mean(unlist(knnAccuracy))
cat("\n")
print("KNN Accuracy Measures(Recall,Precision,F1) :")
# print(KnnResults)
barKnn<-KnnResults[1:6,1:6]
print("Key- 0:Africa,1:Asia,2:Australia,3:Europe,4:North America,5:South America")
print("Accuracy Measures for first group")
print(barKnn)

print("*************************")
cat("\n")


#SVM
print("SVM Accuracy for the dataset groups:")
# print(table(svmAccuracy))
print("Best Accuracy obtained using SVM :")
bestAccuracySvm<-max(unlist(svmAccuracy))
print(bestAccuracySvm)
meanAccuracySVM<-mean(unlist(svmAccuracy))
cat("\n")
print("SVM Accuracy Measures(Recall,Precision,F1) :")
#print(svmResults)
barSvm<-svmResults[1:6,1:6]
print("Key- 0:Africa,1:Asia,2:Australia,3:Europe,4:North America,5:South America")
cat("\n")
print("Accuracy Measures for first group")
print(barSvm)

print("*************************")
cat("\n")


#RIPPER
print("RIPPER Accuracy for the dataset groups:")
print(table(ripperAccuracy))
print("Best Accuracy obtained using RIPPER :")
bestAccuracyRipper<-max(unlist(ripperAccuracy))
print(bestAccuracyRipper)
meanAccuracyRip<-mean(unlist(ripperAccuracy))
print("RIPPER Accuracy Measures(Recall,Precision,F1) :")
#print(ripperResults)
barRip<-ripperResults[1:6,1:6]
print("Key- 0:Africa,1:Asia,2:Australia,3:Europe,4:North America,5:South America")
cat("\n")
print("Accuracy Measures for first group")
print(barRip)

print("*************************")
cat("\n")


#C4.5
print("C4.5 Accuracy for the dataset groups:")
print(table(c45Accuracy))
#Accuracy Measures 
meanAccuracyC45<-mean(unlist(c45Accuracy))

print("Best Accuracy obtained using C4.5 :")
bestAccuracyC45<-max(unlist(c45Accuracy))
print(bestAccuracyC45)
cat("\n")
print("C4.5 Accuracy Measures(Recall,Precision,F1) :")
#print(c45Results)
barC45<-c45Results[1:6,1:6]
print("Accuracy Measures for first group")
print(barC45)
cat("\n")
print("Key- 0:Africa,1:Asia,2:Australia,3:Europe,4:North America,5:South America")
cat("\n")



print("Comparison of Accuracies for the models:")
compareAccuracy=data.frame(KNN=bestAccuracyKnn,SVM=bestAccuracySvm,C45=bestAccuracyC45,RIPPER=bestAccuracyRipper)
print(compareAccuracy)

print("*************************")









