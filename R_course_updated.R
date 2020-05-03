library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(class)
library(randomForest)
library(tree)
orchid <- read.table(url("https://gist.githubusercontent.com/CptnCrumble/e01af3b83ffc463f4bb5776d0213f14b/raw/5382eee21b6e5b796541fd8053c5f733fd6eb9c7/orchids.txt"))
attach(orchid)



#Plotting the boxplots of the data 
boxplot(X1 ~ loc, 
        xlab = "Location", 
        ylab = "Petal length",
        col = c("#f54242", "#d4f542", "#42f56f"))

boxplot(X2 ~ loc, 
        xlab = "Location", 
        ylab = "Leaf width",
        col = c("#f54242", "#d4f542", "#42f56f"))

boxplot(X3 ~ loc, 
        xlab = "Location", 
        ylab = "Petal width",
        col = c("#f54242", "#d4f542", "#42f56f"))

aggregate(X1 ~ loc, FUN = mean, 2)
# The mean of the petal length (X1) shows there is difference between the three locations
aggregate(X2 ~ loc, FUN = mean)
# The mean of the leaf width also shows there is a difference between the three locations
aggregate(X3 ~ loc, FUN = mean)
#  The mean of the petal width are very close and cannot be used to differenciate
# between the locations of the data
# From the data provided only the Petal length (X1) and Leaf width (X2) would be
# Reliable to differenciate between the data

# Orchids bivariate scatter plots
ggplot(orchid, aes( x = X1, y = X2)) +
  geom_point(aes (color = factor(loc)), shape = 16, size = 2)+
  theme_classic()+
  labs(title ="Orchids graph", x = "Petal length (mm)", y = "Leaf width (mm)", color = "Orchids Location\n")


# Creating Training data and test data
set.seed(1)
data.subset <- sample(270, 210)
model.train <- orchid[data.subset,]
model.test <- orchid[-data.subset,]
set.seed(1)
##########################################################################
# KNN method

set.seed(1)

model.knn <- train(loc~.-X3, 
                   data = model.train, 
                   method = "knn", 
                   trControl = trainControl(method  = "LOOCV"),
                   preProcess = c("center", "scale"), # Normalize the data
                   tuneLength = 10) # Number of possible K values to evaluate

plot(model.knn)

model.knn$bestTune

predict.knn <- model.knn %>% predict(model.test)

# Plotting the KNN graph
pl = seq(min(model.test$X1), max(model.test$X1), by=0.1)
pw = seq(min(model.test$X2), max(model.test$X2), by=0.1)

# generates the boundaries for the graph
lgrid <- expand.grid(X1=pl, X2=pw, X3=19.73)

knnPredGrid <- predict(model.knn, newdata=lgrid)

knnPredGrid <- model.knn %>% predict(lgrid)

knnPredGrid = as.numeric(knnPredGrid)

predict.knn <- as.numeric(predict.knn)

model.test$loc <- predict.knn

probs <- matrix(knnPredGrid, 
                length(pl), 
                length(pw))

contour(pl, pw, probs, labels="", 
        xlab="Petal length (mm)", ylab="leaf width (mm)", 
        main="K-Nearest Neighbor", axes=T)

gd <- expand.grid(x=pl, y=pw)

points(gd, pch=3, col=probs, cex = 0.1)

# add the test points to the graph
points(model.test$X1, model.test$X2, col= model.test$loc, cex= 2, pch = 20) 


####################################################################################
# Random forest Bagging method
set.seed(1)
bag.tree <- randomForest(loc ~ . -X3, data = orchid, subset = data.subset,
                         mtry = 2, importance = TRUE)
round(importance(bag.tree), 2)

varImpPlot(bag.tree)

bag_predict <- predict(bag.tree, model.test, type = "class")

# Creating plot for Bagging method using base R plot
lgrid <- expand.grid(X1=pl, X2=pw, X3=19.73)

bagPredGrid <- predict(bag.tree, newdata=lgrid)

bagPredGrid <- bag.tree %>% predict(lgrid)

bagPredGrid = as.numeric(bagPredGrid)

predict.bag <- as.numeric(bag_predict)

model.test$loc <- predict.bag

probs <- matrix(bagPredGrid, length(pl), length(pw))

contour(pl, pw, probs, labels="", 
        xlab="Petal length (mm)", ylab="leaf width (mm)", 
        main="Random forest Bagging method", axes=T)

gd <- expand.grid(x=pl, y=pw)

points(gd, pch=3, col=probs)

# add the test points to the graph

points(model.test$X1, model.test$X2, col= model.test$loc, cex= 2, pch = 20)

######################################################################
# Support vector machine
set.seed(1)

tune.out = tune(svm, loc ~ X1 + X2, data = orchid[data.subset,],
                kernel ="linear",
                ranges = list(cost = seq(from = 0.01,to = 2, length = 40) ))

plot(tune.out$performances$cost, tune.out$performances$error)

summary(tune.out)
tune.out$best.model$cost

bestmod = tune.out$best.model
summary(bestmod)

plot(bestmod, data = model.test, X2~X1)

ypred_linear = predict(bestmod, model.test)

# Support vector machine polynomial kernels
set.seed(1)

tune.out_poly = tune(svm, loc ~ X1 + X2, data = orchid[data.subset,],
                     kernel ="polynomial",
                     ranges = list(cost = seq(from = 0.01,to = 3, length = 30)))

plot(tune.out_poly$performances$cost, tune.out_poly$performances$error)

summary(tune.out_poly)

bestmod_poly = tune.out_poly$best.model
summary(bestmod_poly)
tune.out_poly$best.model$cost

plot(bestmod_poly, data = model.test, X2~X1)

ypred = predict(bestmod_poly, model.test)

# Which kernal is more suitable

# Calculating Test Errors
# KNN method
tab <- table(predict.knn,model.test$loc)
tab
1-(sum(tab) - sum(diag(tab))) / sum(tab)

# Random forest bagging method
tab.bag <- table(bag_predict,model.test$loc)
tab.bag
1-(sum(tab.bag) - sum(diag(tab.bag))) / sum(tab.bag)

# Support vector machine Linear
tab.linear <- table(ypred_linear,model.test$loc)
tab
1-(sum(tab.linear) - sum(diag(tab.linear))) / sum(tab.linear)

# Support vector machine Polynomial kernel
tab.poly <- table(ypred,model.test$loc)
tab.poly
1-(sum(tab.poly) - sum(diag(tab.poly))) / sum(tab.poly)

