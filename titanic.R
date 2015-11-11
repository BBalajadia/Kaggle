#################################################
# Bryan R. Balajadia - 30 September 2015
# Titanic: Machine Learning from Disaster
# Method: Conditional Random Forest in R
# Submission result: 0.82297 (rank 39th/3368)
#################################################

Data & Description: https://www.kaggle.com/c/titanic

# Read-in data to R
train <- read.csv("train.csv", sep =",", header = TRUE)
test <- read.csv("test.csv", sep =",", header = TRUE)

# Combine train and test sets
test$Survived <- 0
all <- rbind(train, test)

summary(all)


#---------- HANDLE MISSING DATA ----------#

all$Embarked[!all$Embarked %in% c("C", "Q", "S")] <- "S"      # Fill-in NA Embarked with mode ("S")
all$Embarked <- factor(all$Embarked)        # Drop unused factor level

all$Fare[is.na(all$Fare)] <- median(all$Fare, na.rm = TRUE)      # Fill-in NA Fare with the median value

# Impute NA age via decision tree
library(rpart)
Age.i <- rpart(Age ~ Pclass + Sex + Fare + Embarked + SibSp + Parch, data = all[!is.na(all$Age),], method = "anova")
all$Age[is.na(all$Age)] <- predict(Age.i, all[is.na(all$Age),])

summary(all)
str(all)


#---------- VARIABLES ----------#

## Factor Pclass
all$Pclass <- factor(all$Pclass)

## Create Title variable using the Name info
## Hypothesis: Title is linked to AGE and SOCIAL STATUS 
all$Name <- as.character(all$Name)        # From factor to character
all$Title <- sapply(all$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})  # Get Title from name
all$Title <- sub(' ', '', all$Title)      # Remove beginning space  
all$Title <- factor(all$Title)          

## Family Size
all$FamilySize <- all$SibSp + all$Parch + 1


## Family ID for large families 
## Large families having trouble getting to boats together?
all$Surname <- sapply(all$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]}) # Get surname from Name
all$FamilyID <- paste(as.character(all$FamilySize), all$Surname, sep="") # Combine FamilySize and Surname
all$FamilyID[all$FamilySize <= 2] <- "Small"   # Assigning a value 'Small' to FamilyID
famIDs <- data.frame(table(all$FamilyID))   # Frequency for FamilyID
famIDs <- famIDs[famIDs$Freq <= 2,]         # Subset unexpectedly small-sized FamilyID groups
all$FamilyID[all$FamilyID %in% famIDs$Var1] <- "Small"   #Overwrite any family IDs that were not correctly identified 
all$FamilyID <- as.factor(all$FamilyID)     # factor


#---------- RESPLIT TRAIN AND TEST SET ----------#
train <- all[1:891,]
test <- all[892:1309,]



#---------- MODELING & PREDICTION ----------#

## Conditional Random Forest Model
library(party)
set.seed(1992) # RANDOM Forest!
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Fare + Embarked + Title + FamilySize + FamilyID, data=train, controls=cforest_unbiased(ntree=1501, mtry=3))
pred <- predict(fit, OOB=TRUE, test, type = "response")



#---------- SUBMISSION ----------#

# Create a data frame containing PassengerId & Survived.
my_submission <- data.frame(PassengerId = test$PassengerId, Survived = pred)

# Write solution to a csv file
write.csv(my_submission, file = "my_submission.csv", row.names = FALSE)


# Congrats! You're now in the top 1.16% of submissions!
