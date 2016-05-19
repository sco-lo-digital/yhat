#Scott Jacobs
#adult data set for YHAT demonstrating Caret
# http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
library(tidyr)
#tidy data
adult <- read.csv("~/YHAT/adult.csv", header=FALSE, stringsAsFactors=FALSE)
col_names <- c("age","workclass","fnlwgt", "education","educationnum",
               "maritalstatus","occupation", "relationship", "race",
               "sex", "capitalgain","capitalloss", "hoursperweek",
               "nativecountry", "Target")
names(adult) <- col_names
adult<- adult[, order(sapply(adult, typeof))] #ordered by type
adult$Target <- factor(adult$Target, levels = c(" <=50K"," >50K"), labels = c("U", "O"))
#adult$Target <- factor(adult$Target, levels = c(" <=50K", " >50K"), labels = c("under","over"))
adult[,1:8] <- lapply(adult[,1:8], as.factor) #Create factors
# dumAdult <- dummyVars(~ nativecountry,sep = ".", data = adult)
# dumAdult<- predict(dumAdult, adult) %>% data.frame
# adult <- adult[,-8]
# adult <- cbind(adult, dumAdult)
adult[, 10:15] <- scale(adult[, 10:15]) #center and scale for ML
#adult <- adult[, order(names(adult))] #alphabetically ordered
str(adult)
#inspect data 
#library(Hmisc)
#describe(adult) #tells us Target has a missing value. Let's drop it.
adult <- adult[-which(is.na(adult$Target)),] #Removed observation which has na for target
str(adult)

#load caret
library(caret)
library(AppliedPredictiveModeling)
transparentTheme(trans = .9)
#Show basic plotting features in caret
adult_samp <- adult[sample(nrow(adult), 1000), ]
featurePlot(
  x = adult_samp[, 10:15],
  y = adult_samp$Target,
  plot = "pairs",
  ## Add a key at the top
  auto.key = list(columns = 6)
)


featurePlot(
  x = adult_samp[, 10:15],
  y = adult_samp$Target,
  plot = "density",
  ## Pass in options to xyplot() to
  ## make it prettier
  scales = list(
    x = list(relation = "free"),
    y = list(relation = "free")
  ),
  adjust = 1.5,
  pch = "|",
  layout = c(6, 1),
  auto.key = list(columns = 6)
)

featurePlot(
  x = adult_samp[, 10:15],
  y = adult_samp$Target,
  plot = "box",
  ## Pass in options to bwplot()
  scales = list(y = list(relation = "free"),
                x = list(rot = 90)),
  layout = c(6, 1),
  auto.key = list(columns = 6)
)
##### Method for splitting data, often used to create training data.
set.seed(3456)
trainIndex <- createDataPartition(adult$Target, p = .75,
                                  list = FALSE,
                                  times = 1)
adult_samp <- adult[ trainIndex,]
# Show pre-proc options
#http://topepo.github.io/caret/preprocess.html

#Show options for data spliting, up/down sampling

# Show basic modeling syntax


set.seed(825)
gbmFit1 <- train(Target ~ ., data = adult_samp[,-1:-8],
                 method = "gbm",
                 preProcess = c("center","scale"),  # Center and scale data
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
gbmFit1
confusionMatrix(gbmFit1)
plot(gbmFit1)

#Now make same model with tuning grid
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 3), #maximum is floor(sqrt(NCOL(adult_samp))
                        n.trees = (1:30)*10,
                        shrinkage = 0.15,
                        n.minobsinnode = 20)
# cv
set.seed(123)
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 22)
seeds[[51]] <- sample.int(1000, 1)

fitControl <- trainControl(method = "repeatedcv",
                           repeats = 1,
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)

set.seed(123)
gbmFit2 <- train(Target ~ ., 
                 data = adult_samp[,-1:-8],
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid,
                 metric="ROC")
gbmFit2
confusionMatrix(gbmFit2)
plot(gbmFit2)

svmFit <- train(Target ~ ., data = adult_samp[,-1:-8],
                         method = "svmRadial",   # Radial kernel
                         #tuneLength = 9,					# 9 values of the cost function
                         preProcess = c("center","scale"),  # Center and scale data
                         metric="ROC",
                         trControl=fitControl)

confusionMatrix(svmFit)

rfFit <- train(Target ~ ., data = adult_samp,
               method = "rf",   # Random Forest
               preProc = c("center","scale"),  # Center and scale data
               metric="ROC",
               trControl=fitControl)


resamps <- resamples(list(GBM1 = gbmFit1,
                          GBM2 = gbmFit2,
                          SVM = svmFit,
                          RF = rfFit))
resamps
summary(resamps)

bwplot(resamps, layout = c(2, 1))

