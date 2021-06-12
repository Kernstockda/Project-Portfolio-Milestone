#install.packages("e1071")
#install.packages("naivebayes")
library(e1071)
library(naivebayes)
library(ggplot2)
library(tidyverse)
setwd("C:/Users/Drfu/OneDrive - Syracuse University/syracus/IST707/Final project")

# Read in .csv data
filename="diabetes.csv"
df_raw <- read.csv(filename, header = TRUE, na.strings = "NA")
(head(df_raw))  # check data frame
(str(df_raw))  # check structure


#Data cleaning
Data_tem <- subset(df_raw,select = -c(Pregnancies,Outcome)) # delete label
Data_tem[Data_tem==0]<-NA  # replace 0 to NA
sum(is.na(Data_tem))  # Count NA
Data_tem$label<-df_raw$Outcome  # add label back to df 
Data_tem$Pregnancies<-df_raw$Pregnancies  # add label back to df 
str(Data_tem)
Data_tem1<-na.omit(Data_tem)  # remove rows with NA
any(is.na(Data_tem1))  # check any NA
Data_clean<-Data_tem1  # rename
str(Data_clean)



# Data transformation
# convert label to factor
Data_clean$label <- as.factor(Data_clean$label)


#EXPLORE data
head(Data_clean)
(table(Data_clean$label))  # frequency
(plot_glu<-ggplot(Data_clean,aes(x=Glucose))+geom_histogram(aes(fill=Glucose),color="green",binwidth=20))+ggtitle("distribution for Glucose")  # distribution for Glucose 
(plot_bp<-ggplot(Data_clean,aes(x=BloodPressure))+geom_histogram(aes(fill=BloodPressure),color="green",binwidth=5))+ggtitle("distribution for BloodPressure ")  # distribution for BloodPressure 
(plot_st<-ggplot(Data_clean,aes(x=SkinThickness))+geom_histogram(aes(fill=SkinThickness),color="green",binwidth=5))+ggtitle("distribution for SkinThickness ")  # distribution for SkinThickness 
(plot_ins<-ggplot(Data_clean,aes(x=Insulin))+geom_histogram(aes(fill=Insulin),color="green",binwidth=50))+ggtitle(" distribution for Insulin")  # distribution for Insulin 
(plot_bmi<-ggplot(Data_clean,aes(x=BMI))+geom_histogram(aes(fill=BMI),color="green",binwidth=5))+ggtitle(" distribution for BMI")  # distribution for BMI 
(plot_dpf<-ggplot(Data_clean,aes(x=DiabetesPedigreeFunction))+geom_histogram(aes(fill=DiabetesPedigreeFunction),color="green",binwidth=0.1))+ggtitle("distribution for DiabetesPedigreeFunction ")  # distribution for DiabetesPedigreeFunction 
(plot_age<-ggplot(Data_clean,aes(x=Age))+geom_histogram(aes(fill=Age),color="green",binwidth=5))+ggtitle(" distribution for Age ")  # distribution for Age 
(plot_pre<-ggplot(Data_clean,aes(x=Pregnancies))+geom_histogram(aes(fill=Pregnancies),color="green",binwidth=1))+ggtitle("distribution for Pregnancies ")  # distribution for Pregnancies 


# partitioning data
# allocate 80% to training and remainder to testing
total_rows <- nrow(Data_clean)

# training rows = 80%
train_rows <- total_rows * 0.8

#testing rows = 20%
test_rows <- total_rows * 0.2

# generate train and test datasett
df <- Data_clean[sample(nrow(Data_clean)), ]  #sample rows 
train_df <- df[1:train_rows, ]                #get training set
test_df <- df[(train_rows+1):total_rows, ]    #get test set

# view the datasets
str(train_df)
str(test_df)
(dim(train_df))
(dim(test_df))
# do we have a good ratio of both outcomes in both training and test data?
table(train_df$label)
table(test_df$label)

# keep a record of just the labels in the test data
test_justLabel <- test_df$label

# remove labels out of the testing data
test_noLabel <- test_df[-c(8)]



# view both
head(test_noLabel)
head(test_justLabel)



# MODEL 1
### NAIVE BAYES ALGORITHM
### USING NAIVE BAYES PACKAGE
NB1<- naive_bayes(train_df$label~., data=train_df)
pred_NB1<-predict(NB1, test_noLabel, type = c("class"))
head(predict(NB1, test_noLabel, type = "prob"))
(Ptable_NB1<-table(pred_NB1,test_justLabel)) # confusion tale
plot(NB1, legend.box = FALSE, arg.num = list())
(accuracy_NB1 <- sum(diag(Ptable_NB1))/sum(Ptable_NB1))  # accuracy for NB


###DECISION TREE ALGORITHM
######################################### BUILD Decision Trees ----------------------------
#install.packages("rpart")
#install.packages('rattle')
#install.packages('rpart.plot')
#install.packages('RColorBrewer')
#install.packages("Cairo")
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(Cairo)

# MODEL 2.1
## Create the decision tree using rpart
tree1 <- rpart(train_df$label ~ ., data = train_df, method="class")
summary(tree1)
pred_tree1= predict(tree1,test_noLabel, type="class")
(head(pred_tree1,n=10))
(head(test_df$label, n=10))

(ptable_tree1<-table(pred_tree1,test_justLabel)) # confusion tale
(accuracy_tree <- sum(diag(ptable_tree1))/sum(ptable_tree1))  # accuracy for tree1
fancyRpartPlot(tree1)
text(tree1) #  plot text

# MODEL 2.2
## Let's reduce the tree size
tree2 <- rpart(label ~ Glucose + Insulin+ BMI+Age,
               data=train_df,
               method="class", 
               control=rpart.control(minsplit=60, cp=0.001))
fancyRpartPlot(tree2)
pred_tree2= predict(tree2,test_noLabel, type="class")


(ptable_tree2<-table(pred_tree2,test_justLabel)) # confusion tale
(accuracy_tree2 <- sum(diag(ptable_tree2))/sum(ptable_tree2))  # accuracy for tree2
fancyRpartPlot(tree2)
text(tree2) #  plot text


# MODEL 3.1  Polynomial Kernel...

SVM_p <- svm(label ~., data=train_df,
             kernel="polynomial", cost=.1, 
             scale=FALSE)
##Prediction --
pred_SVM_p <- predict(SVM_p,test_noLabel, type="class")
(ptable_SVM_p<-table(pred_SVM_p,test_justLabel)) # confusion tale
(accuracy_SVM_p <- sum(diag(ptable_SVM_p))/sum(ptable_SVM_p))  # accuracy for SVM_p

# MODEL 3.2  linear Kernel...

SVM_l <- svm(label ~., data=train_df,
             kernel="linear", cost=.1, 
             scale=FALSE)
##Prediction --
pred_SVM_l <- predict(SVM_l,test_noLabel, type="class")
(ptable_pred_SVM_l<-table(pred_SVM_l,test_justLabel)) # confusion tale
(accuracy_SVM_l <- sum(diag(ptable_pred_SVM_l))/sum(ptable_pred_SVM_l))  # accuracy for pred_SVM_l

# MODEL 3.3  radial Kernel...

SVM_r <- svm(label ~., data=train_df,
             kernel="radial", cost=.1, 
             scale=FALSE)
##Prediction --
pred_SVM_r <- predict(SVM_r,test_noLabel, type="class")
(ptable_pred_SVM_r<-table(pred_SVM_r,test_justLabel)) # confusion tale
(accuracy_SVM_r <- sum(diag(ptable_pred_SVM_r))/sum(ptable_pred_SVM_r))  # accuracy for pred_SVM_r

#################   MODEL 4 Set up the RANDOM FOREST -----------------

library(randomForest)
rf<- randomForest(label~., data=train_df,ntree=100,proximity=TRUE)
pred_rf <-predict(rf)

## see accuracy
(ptable_rf <- table(pred_rf, train_df$label))  # confusion tale
(accuracy_rf <- sum(diag(ptable_rf))/sum(ptable_rf))   # accuracy Rate for RF.
plot(rf)


############################
df_result<-data.frame(accuracy=c(accuracy_NB1, accuracy_tree,accuracy_tree2,accuracy_SVM_p,accuracy_SVM_l,accuracy_SVM_r,accuracy_rf))
str(df_result)
df_result

b<-barplot(df_result$accuracy,main = "Accuracy",col = c("red"),
           names.arg=c("NB1","tree1","tree2","svm_p","svm_l","svm_r","RF"))



