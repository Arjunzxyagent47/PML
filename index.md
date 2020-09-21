# Practical Machine Learning Course Project

## Overview

<i>One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise

## Import Libraries


```R
set.seed(1234)
```


```R
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(randomForest)
library(corrplot)
library(gbm)

```

## Data Pre-Processing

<h3> Source </h3>

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **“Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13)”.** Stuttgart, Germany: ACM SIGCHI, 2013.

<h3> Data Loading


```R
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
training <- read.csv(url(UrlTrain))
testing  <- read.csv(url(UrlTest))


# create a partition with the training dataset 
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
dim(TrainSet)
```


<ol class=list-inline>
	<li>13737</li>
	<li>160</li>
</ol>




```R
dim(TestSet)
```


<ol class=list-inline>
	<li>5885</li>
	<li>160</li>
</ol>



<h3> Data Cleaning

 Remove Variables with Nearly Zero Variance


```R
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
dim(TrainSet)
```


<ol class=list-inline>
	<li>13737</li>
	<li>103</li>
</ol>




```R
dim(TestSet)
```


<ol class=list-inline>
	<li>5885</li>
	<li>103</li>
</ol>



Clear out variables with mostly NAs


```R
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TrainSet)
```


<ol class=list-inline>
	<li>13737</li>
	<li>59</li>
</ol>




```R
dim(TestSet)
```


<ol class=list-inline>
	<li>5885</li>
	<li>59</li>
</ol>



Remove identification only variables (col 1-5)


```R
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
```


<ol class=list-inline>
	<li>13737</li>
	<li>54</li>
</ol>




```R
dim(TestSet)
```


<ol class=list-inline>
	<li>5885</li>
	<li>54</li>
</ol>



Cleaning has helped us reduce the number of variables to 54.

## Correlation Analysis


```R
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```


![png](output_23_0.png)


## Building Prediction Models

## Using Random Forests


```R
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",trControl=controlRF)
modFitRandForest$finalModel
```


    
    Call:
     randomForest(x = x, y = y, mtry = param$mtry) 
                   Type of random forest: classification
                         Number of trees: 500
    No. of variables tried at each split: 27
    
            OOB estimate of  error rate: 0.23%
    Confusion matrix:
         A    B    C    D    E  class.error
    A 3905    0    0    0    1 0.0002560164
    B    7 2648    2    1    0 0.0037622272
    C    0   10 2386    0    0 0.0041736227
    D    0    0    7 2245    0 0.0031083481
    E    0    0    0    4 2521 0.0015841584



```R
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1674    3    0    0    0
             B    0 1135    5    0    0
             C    0    1 1021    4    0
             D    0    0    0  959    0
             E    0    0    0    1 1082
    
    Overall Statistics
                                             
                   Accuracy : 0.9976         
                     95% CI : (0.996, 0.9987)
        No Information Rate : 0.2845         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.997          
                                             
     Mcnemar's Test P-Value : NA             
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            1.0000   0.9965   0.9951   0.9948   1.0000
    Specificity            0.9993   0.9989   0.9990   1.0000   0.9998
    Pos Pred Value         0.9982   0.9956   0.9951   1.0000   0.9991
    Neg Pred Value         1.0000   0.9992   0.9990   0.9990   1.0000
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2845   0.1929   0.1735   0.1630   0.1839
    Detection Prevalence   0.2850   0.1937   0.1743   0.1630   0.1840
    Balanced Accuracy      0.9996   0.9977   0.9970   0.9974   0.9999



```R
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest -> Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
```


![png](output_28_0.png)


## Using Decision Trees


```R
# model fit
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
#fancyRpartPlot(modFitDecTree)
rpart.plot(modFitDecTree)
```

    Warning message:
    "labs do not fit even at cex 0.15, there may be some overplotting"


![png](output_30_1.png)



```R
# prediction on Test dataset
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1539  214   36   70   82
             B   68  690   87   86  124
             C    5   86  818  132   63
             D   42   83   57  576   39
             E   20   66   28  100  774
    
    Overall Statistics
                                              
                   Accuracy : 0.7472          
                     95% CI : (0.7358, 0.7582)
        No Information Rate : 0.2845          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.6782          
                                              
     Mcnemar's Test P-Value : < 2.2e-16       
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9194   0.6058   0.7973  0.59751   0.7153
    Specificity            0.9045   0.9231   0.9411  0.95509   0.9554
    Pos Pred Value         0.7929   0.6540   0.7409  0.72271   0.7834
    Neg Pred Value         0.9658   0.9070   0.9565  0.92374   0.9371
    Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
    Detection Rate         0.2615   0.1172   0.1390  0.09788   0.1315
    Detection Prevalence   0.3298   0.1793   0.1876  0.13543   0.1679
    Balanced Accuracy      0.9119   0.7644   0.8692  0.77630   0.8354



```R
# plot matrix results
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree -> Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
```


![png](output_32_0.png)


## Using GBMs


```R
# model fit
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```


    A gradient boosted model with multinomial loss function.
    150 iterations were performed.
    There were 53 predictors of which 53 had non-zero influence.



```R
# prediction on Test dataset
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1674    3    0    2    0
             B    0 1124    6    4    1
             C    0   12 1017   10    0
             D    0    0    3  945    5
             E    0    0    0    3 1076
    
    Overall Statistics
                                             
                   Accuracy : 0.9917         
                     95% CI : (0.989, 0.9938)
        No Information Rate : 0.2845         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.9895         
                                             
     Mcnemar's Test P-Value : NA             
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            1.0000   0.9868   0.9912   0.9803   0.9945
    Specificity            0.9988   0.9977   0.9955   0.9984   0.9994
    Pos Pred Value         0.9970   0.9903   0.9788   0.9916   0.9972
    Neg Pred Value         1.0000   0.9968   0.9981   0.9961   0.9988
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2845   0.1910   0.1728   0.1606   0.1828
    Detection Prevalence   0.2853   0.1929   0.1766   0.1619   0.1833
    Balanced Accuracy      0.9994   0.9923   0.9934   0.9893   0.9969



```R
# plot matrix results
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM -> Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))
```


![png](output_36_0.png)


## Which Model to Use?

The accuracy of the 3 modeling methods above are:

**Random Forest : 99.76%** 

**Decision Tree : 74.72%**
 
**GBM : 98.17%**

<h3>Hence, we'll go with the Random Forest model

## Applying the Selected Model to the Test Data

**The Random Forest model will be applied to predict the 20 quiz results**


```R
predictTEST <- predict(modFitRandForest, newdata=testing)
predictTEST
```


<ol class=list-inline>
	<li>B</li>
	<li>A</li>
	<li>B</li>
	<li>A</li>
	<li>A</li>
	<li>E</li>
	<li>D</li>
	<li>B</li>
	<li>A</li>
	<li>A</li>
	<li>B</li>
	<li>C</li>
	<li>B</li>
	<li>A</li>
	<li>E</li>
	<li>E</li>
	<li>A</li>
	<li>B</li>
	<li>B</li>
	<li>B</li>
</ol>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<ol class=list-inline>
		<li>'A'</li>
		<li>'B'</li>
		<li>'C'</li>
		<li>'D'</li>
		<li>'E'</li>
	</ol>
</details>

