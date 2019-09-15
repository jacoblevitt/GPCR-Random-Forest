library('MultivariateRandomForest')

dataX_train <- read.csv('dataX_train.csv')
dataY_train <- read.csv('dataY_train.csv')
dataX_test <- read.csv('dataX_test.csv')

dataX_train <- data.matrix(dataX_train)
dataY_train <- data.matrix(dataY_train)
dataX_test <- data.matrix(dataX_test)

build_forest_predict(dataX_train, dataY_train, 20, 1000, 100, dataX_test)


