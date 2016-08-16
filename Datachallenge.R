#install.packages('caret')
#install.packages("https://h2o-release.s3.amazonaws.com/h2o-ensemble/R/h2oEnsemble_0.1.8.tar.gz", repos = NULL)
#To install h2o
## The following two commands remove any previously installed H2O packages for R.
#if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
#if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# Next, we download, install and initialize the H2O package for R.
#install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1497/R", getOption("repos"))))
#install.packages('Metrics')
library(Metrics)
library(mice)
library(VIM)
#install.packages('outliers')
library(outliers)
library(h2oEnsemble)
library(h2o)
library(caret)

setwd("~/Documents/AerialIntel")

#Read CSV file
wheat.2013<-read.csv('wheat-2013-supervised.csv')
wheat.2014<-read.csv('wheat-2014-supervised.csv')

#Check number of NAs and other descriptives to get a sense of data
summary(wheat.2013)
summary(wheat.2014)

#Combining data from 2013 and 2014 
wheat.df<-rbind(wheat.2014,wheat.2013)
summary(wheat.df)

#Removing features that are to be removed based on the question!
wheat.df$CountyName=NULL
wheat.df$State=NULL
wheat.df$Date=NULL
wheat.df$Latitude=NULL
wheat.df$Longitude=NULL
wheat.df$precipTypeIsOther=NULL  #Constant Column

#Missing values Percentage calculation
sum(is.na(wheat.df))/prod(dim(wheat.df))*100  
#Missing values are < 1% so we can neglect the imputation of the missingvalues as we need not induce bias via imputed values.
wheat.df<-na.omit((wheat.df))


#split Train and Test dataset
set.seed(3456)
trainIndex <- createDataPartition(wheat.df$Yield, p = .8, list = FALSE, times = 2)
wheat.df_Train <- wheat.df[ trainIndex,]
wheat.df_Test  <- wheat.df[-trainIndex,]


#Scale data to make all feature equally important independant of their ranges
#Scale after splitting to avoid data snooping
wheat.df_Train <- as.data.frame(scale(wheat.df_Train))
wheat.df_Test <- as.data.frame(scale(wheat.df_Test))

#Exploratory data analysis
#Outlier detection
#Boxplots
qplot(Yield,pressure,data=wheat.df_Train,geom='boxplot', main = 'Box Plot of Tip vs Distance')
plot(wheat.df_Test)
data.outliers<-outlier(wheat.df_Train$Yield)
#These ouliers cannot be removed as they donot seem to be an error in measurement.


#Correlation Plots
#install.packages('corrplot')
library(corrplot)
M <- cor(wheat.df_Train)
corrplot(M, method="circle")
corrplot(M, order="hclust", addrect=2) # Correlation plot of Hierachial Clusters

#Predictive Modeling!

#Using H2o package for the predictive modelling as it has capabilities to run the models in a 
#distributed fashion on the JVM using in memory processing.

## Create an H2O cloud 
h2o.init(nthreads=-1,max_mem_size = "10G") ## -1: use all available threads
h2o.removeAll() # Clean slate - just in case the cluster was already running
h2o.train<-as.h2o(wheat.df_Train) 
h2o.test<-as.h2o(wheat.df_Test)  

head(h2o.train)
y <- c("Yield")
x <- setdiff(names(h2o.train), y)


#first predictive model
rf1 <- h2o.randomForest(         
  training_frame = h2o.train,              ## the H2O frame for training
  x=x,                                     ## the predictor columns, by column index
  y=y,                                     ## the target index (what we are predicting)
  model_id = "rf_covType_v1",              ## name the model in H2O
  ntrees = 150,                            ## use a maximum of 200 trees to create the model
  score_each_iteration = F,    
 
  nfolds=5,
  seed = 1000000,
  stopping_rounds = 3,
  stopping_tolerance = 0.001)            

#Predict the values on test data
Rf_predictions<-h2o.predict(object = rf1,newdata = h2o.test)
performance<-h2o.performance(rf1,newdata = h2o.test)
h2o.mse(performance)
wheat.df_Test$YieldPredicted=Rf_predictions

#Calculate the RMSE
rmse(as.vector(wheat.df_Test$Yield),as.vector(wheat.df_Test$YieldPredicted))


#From Variable importances we find variables that are not contributing much to the model but increasing the complexity of the model

gc() #Garbage Collecton to freed heap space!
gbm1 <- h2o.gbm(
  training_frame = h2o.train,     ##
  x=x,                     ##
  y=y,    
  nfolds = 3,
  stopping_metric = "MSE",
  ntrees = 100,                ## decrease the trees, mostly to allow for run time
  learn_rate = 0.2,           ## increase the learning rate (from 0.1)
  max_depth = 60,             ## increase the depth (from 5)
  stopping_rounds = 3,        ## 
  stopping_tolerance = 0.01,  ##
  score_each_iteration = T,   ##
  model_id = "gbm_covType2",  ##
  seed = 20000
)  

#Predict the values on test data
gbm_predictions<-h2o.predict(
  object = gbm1,newdata = h2o.test)
h2o.performance(gbm1,newdata = h2o.test)

#Calculate the RMSE of the predictions on the test set
rmse(as.vector(wheat.df_Test$Yield),as.vector(gbm_predictions))
# Approx: 0.42 


#Neural Networks
Dl_model <- h2o.deeplearning(x = x, y = y , model_id = "dll",training_frame = h2o.train,activation ='RectifierWithDropout',
                             epochs =200 ,hidden = c(200,200,150) )

dl_predictions<-h2o.predict(
  object = Dl_model,newdata = h2o.test)
h2o.performance(Dl_model,newdata = h2o.test)

#################
#Elastic Net regression.
h2o.glm(y = y, x = x, training_frame = h2o.train, lambda_search = TRUE, max_active_predictors = 20)



#Super Learning / Ensemble learning with metalearner
####################################################################################
#Custom Wrappers for Base Learners

h2o.gbm <- function(...,    stopping_metric = "MSE",score_each_iteration = F,stopping_tolerance = 0.01,stopping_rounds = 3,nfolds = 3,learn_rate = 0.2, ntrees = 100, seed = 14123) h2o.gbm.wrapper(..., stopping_tolerance=stopping_tolerance,stopping_rounds=stopping_rounds,learn_rate=learn_rate,score_each_iteration=score_each_iteration,ntrees = ntrees, seed = seed)

h2o.rf <- function(...,    stopping_metric = "MSE",score_each_iteration = F,stopping_tolerance = 0.01,stopping_rounds = 3,nfolds = 4,learn_rate = 0.2, ntrees = 150, seed = 14123) h2o.randomForest.wrapper(..., stopping_tolerance=stopping_tolerance,stopping_rounds=stopping_rounds,learn_rate=learn_rate,score_each_iteration=score_each_iteration,ntrees = ntrees, seed = seed)

#####################################################################################


learner <- c("h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper")
metalearner <- "h2o.gbm.wrapper"


# Train the ensemble using 5-fold CV to generate level-one data
# More CV folds will take longer to train, but should increase performance
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = h2o.train, 
                    
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5, shuffle = TRUE))
#Generate predictions for test set
pred <- predict(fit, h2o.test)

#Calculate the RMSE of the predictions for the test set
rmse(as.vector(pred$pred),wheat.df_Test$Yield)

#################################################################################
#Grid Search if time permits

# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.2) 
max_depth_opt <- c(3, 4, 5)
sample_rate_opt <- c(0.7, 0.8, 0.9)
col_sample_rate_opt <- c(0.3, 0.4, 0.5, 0.6)
ntrees<-c(200,100)
nfolds <- 1
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt, 
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt,ntrees=ntrees)
gbm_grid <- h2o.grid("gbm", x = x, y = y,
                     training_frame = h2o.train,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params
)
summary(gbm_grid)
gbm_models[1]


best_model <- h2o.getModel(gbm_grid@model_ids[[1]])
best_model_params<-(best_model@allparameters)
bestgbm_modelid<-best_model_params$model_id
ntreesgbm=best_model_params$ntrees
ntreesgbm=100
learnrategbm=best_model_params$learn_rate
max_depthgbm=best_model_params$max_depth
samplerategbm=best_model_params$sample_rate
colsamplerategbm=best_model_params$col_sample_rate

h2o.gbm.best <- function(..., ntrees = ntreesgbm,learn_rate=learnrategbm,sample_rate=samplerategbm,col_sample_rate = colsamplerategbm,max_depth = max_depthgbm, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees,max_depth = max_depth, col_sample_rate = col_sample_rate, learn_rate=learn_rate,sample_rate=sample_rate,seed = seed)

#RFGS
# RF Hyperparamters
mtries_opt <- 5:20 
max_depth_opt <- c(5, 10, 15, 20, 25)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_per_tree_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(mtries = mtries_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt
                     ,ntrees=ntrees
)

rf_grid <- h2o.grid("randomForest", x = x, y = y,
                    training_frame = h2o.train,
                    
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,                    
                    hyper_params = hyper_params
)
best_modelrf <- h2o.getModel(rf_grid@model_ids[[1]])
best_model_paramsrf<-(best_modelrf@allparameters)
bestgrf_modelid<-best_model_paramsrf$model_id
ntreesrf=best_model_paramsrf$ntrees

max_depthrf=best_model_paramsrf$max_depth
samplerategbm=best_model_paramsrf$sample_rate
mtriesrf=best_model_paramsrf$mtries
h2o.randomForest.4 <- function(..., ntrees =ntreesrf,sample_rate=samplerategbm,mtries=mtriesrf, max_depth=max_depthrf, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate,max_depth = max_depth, seed = seed,mtries = mtries)

#DLGS

# Deeplearning Hyperparamters
activation_opt <- c("Rectifier", "RectifierWithDropout", 
                    "Maxout", "MaxoutWithDropout") 
hidden_opt <- list(c(80,30), c(50,35), c(50,30,10))
l1_opt <- c(0, 1e-3, 1e-5)
l2_opt <- c(0, 1e-3, 1e-5)
epochs = c(20,100)
hyper_params <- list(activation = activation_opt,
                     hidden = hidden_opt,
                     l1 = l1_opt,
                     l2 = l2_opt,
                     epochs =epochs)
gc()

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    training_frame = h2o.train,
                    
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,                    
                    hyper_params = hyper_params)

summary(dl_grid)

best_modeldl <- h2o.getModel(dl_grid@model_ids[[1]])
best_model_paramsdl<-(best_modeldl@allparameters)
bestgdl_modelid<-best_model_paramsdl$model_id
bestdl_act=best_model_paramsdl$activation
bestdl_nfolds=best_model_paramsdl$nfolds
bestdl_epochs=best_model_paramsdl$epochs
bestdl_l1=best_model_paramsdl$l1
bestdl_l2=best_model_paramsdl$l2
bestdl_hidden=best_model_paramsdl$hidden

h2o.deeplearning.4 <- function(..., hidden = bestdl_hidden, activation = bestdl_act, epochs = bestdl_epochs,l2=bestdl_l2,l1=bestdl_l2,nfolds=bestdl_nfolds, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation,l1=l1,l2=l2,epochs = epochs, seed = seed)


metalearner <- "h2o.gbm.wrapper"

learner <- c("h2o.randomForest.4","h2o.gbm.best")

gc()
fit <- h2o.ensemble(x = x, y = y,
                    training_frame = h2o.train,
                    
                    learner = learner,
                    metalearner = metalearner,
                    cvControl = list(V = 5, shuffle = TRUE))

h2o.ensemble_performance(fit,h2o.train)
pp <- predict(fit, h2o.test)

################################################


