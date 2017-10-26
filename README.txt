Readme

Mugdha Mathkar
UFID:54147979

*****Important:Please save the .csv file on disk.You will have to provide the path of the csv file on line number 12 :

Here is the line:
Data_csv <- read.csv("~/Data_original.csv", colClasses=c(NA,"NULL",NA,NA,NA,NA), header = TRUE)
This path has to be set to the working directory.

****************************************************************************
Packages to be installed:
library(caret)
library(e1071) #SVM
library(ggplot2)
library(rJava)
library(RWeka)
library(ROCR)
***************************************************************************
Steps to run the Script:

Run the source() command on the classification.R script.
You can run the following command:
  source('~/classification.R')
Please note:You might have to give the path of the folder where the file is saved.

All the results of the classification models will be displayed on the output screen.Plots will be generated on the Plots dialog box for each classification.

***************************************************************************

To check for specfic values,please use the following variables:

1.Data_original.csv:This is the CSV file used for classification.This has to be placed along with the Rscript.
  
2.compareAccuracy :This variable contains the best accuracies for each model

> compareAccuracy
        KNN       SVM       C45    RIPPER
1 0.5581395 0.5813953 0.5813953 0.6046512

3.KnnResults:This holds all the evaluation measures for the predictions on test sets for each group of training and test set

> KnnResults
         acc f1Measure    recall      prec   microf1 x     continent
1  0.5348837 0.8181818 0.8181818 0.8181818 0.5348837 0        Africa
2  0.5348837 0.4545455 0.4166667 0.5000000 0.5348837 0          Asia
3  0.5348837 0.0000000 0.0000000 0.0000000 0.5348837 0     Australia
4  0.5348837 0.6923077 0.6000000 0.8181818 0.5348837 0        Europe
5  0.5348837 0.0000000 0.0000000 0.0000000 0.5348837 0 North America
6  0.5348837 0.0000000 0.0000000 0.0000000 0.5348837 0 South America

