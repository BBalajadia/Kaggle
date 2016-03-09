#################################################
# Bryan R. Balajadia - 9 March 2016
# Kaggle: Santander Customer CLassification
# Method: XGBoost (Extreme Gradient Boosting)
# Submission result: 0.839319 (248th/805)
#################################################

# Set the working directory
setwd("C:/Users/136241/Desktop/KAGGLE/Santander")

library(xgboost)
library(Matrix)


# Read-in data to R
raw.train <- read.csv("train.csv", sep =",", header = TRUE, stringsAsFactors = FALSE)
raw.test <- read.csv("test.csv", sep =",", header = TRUE, stringsAsFactors = FALSE)

## Combine train and test data
raw.test$TARGET <- -1
all <- rbind(raw.train,raw.test)


#---------- Data Cleaning and Feature Engineering ----------#

## Convert data types and remove constant features!
feature.names <- names(raw.train)[-1]

for (i in feature.names) {
  if (class(all[[i]]) == "numeric") {
    all[[i]] <- all[[i]] / max(all[[i]])
  } else if (class(all[[i]]) == "integer") {
    u <- unique(all[[i]])
    if(length (u) == 1) {
      cat(i, "is a constant feature. Delete! \n")
      all[[i]] <- NULL
    } else if (length (u) < 100) {
      all[[i]] <- factor(all[[i]])
    }
  }
}


## Resplit the data
train <- all[all$TARGET != -1,]
test <- all[all$TARGET == -1,]

## Cleanup useless objects from the environment
rm(all, raw.train, raw.test, feature.names, i, u)

## Removing IDs
train$ID <- NULL
train.y <- train$TARGET
test.id <- test$ID
test$ID <- NULL


#---------- MODELING & PREDICTION: Extreme Gradient Boosting ----------#

set.seed (1992)

## Convert your train into numeric data types
## because XGBoost only works for numeric vectors
## “sparse.model.matrix” is the command to use
train <- sparse.model.matrix(TARGET ~ ., data=train) # ignore 'TARGET'

##------------ Tune and Run the model

## xgb.train takes an xgb.DMatrix as the input.
dtrain <- xgb.DMatrix(data=train, label = train.y)

## For validation set monitoring during training
watchlist <- list(train = dtrain)

param <- list (booster = "gbtree", # general parameter: treebased
               
               objective = "binary:logistic", # learning task parameter
               
               eval_metric = "auc", # assigned according to the objective parameter
               
               eta = 0.3, # set to default value
               # range: [0,1]; Low is more robust to overfitting
               
               max_depth = 15, # default is 6
               # range: [1,infinity)
               
               subsample = 0.5, # Set to 0.5 to prevent overfitting
               # Default is 1; range: [0,1]
               
               colsample_bytree = 0.5 # Subsample ratio of columns when constructing the tree
               # Default is 1
)

clf <- xgb.train (params = param, # list of parameters
                  data = dtrain,
                  nrounds = 350, # number of iterations
                  verbose = 1, # to print information of performance
                  watchlist = watchlist,
                  maximize = FALSE
)


##------------ Compute feature importance matrix
names <- dimnames(train)[[2]]
importance_matrix <- xgb.importance(names, model = clf)
head(importance_matrix)

## Create the graph for the first 10 features!
xgb.plot.importance(importance_matrix[1:10,])


##------------ Predict the values in test set
test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

pred <- predict(clf, test)

#---------- SUBMISSION ----------#

## Saving submission file
submission <- data.frame(ID=test.id, TARGET = pred)
write.csv(submission, "submission.csv", row.names = FALSE)


## Remove all objects
rm(list=ls())
