# Jaime Castaneda
# Linear Regression

luka = read.csv("LukaDoncic.csv")
head(luka)
luka = luka[, -c(1, 2, 4, 5, 6, 9, 12, 22)]

set.seed(123)

sample = sample(c(TRUE, FALSE), nrow(luka), replace = TRUE, prob = c(0.7, 0.3))
train = luka[sample, ]
test = luka[!sample, ]


# splitting
x_train = as.matrix(train[, -which(colnames(train) == "PTS")])
y_train = train$PTS
x_test = as.matrix(test[, -which(colnames(test) == "PTS")])
y_test = test$PTS


# ridge regression
library(glmnet)
fit_ridge = cv.glmnet(x_train, y_train, alpha = 0, nfolds = 7)
plot(fit_ridge)

fit_ridge$lambda.min
fit_ridge$lambda.1se

# training
train_ridge_pred = predict(fit_ridge, x_train, s = "lambda.1se")
rmse_ridge_train = sqrt(mean((train_ridge_pred - y_train)^2))
print(rmse_ridge_train)

ssr_train = sum((train_ridge_pred - y_train)^2)
tss_train = sum((y_train - mean(y_train))^2)
r2_train = 1 - (ssr_train / tss_train)
print(r2_train)

# testing
ridge_pred = predict(fit_ridge, x_test, s = "lambda.1se")
rmse_ridge = sqrt(mean((ridge_pred - y_test)^2))
print(rmse_ridge)
ridge_ratio = rmse_ridge / rmse_ridge_train
print(ridge_ratio)


# linear regression
fullFit = lm(PTS ~ ., data = train)
summary(fit_lm)

# training
train_residuals = fullFit$residuals
train_rmse = sqrt(mean(train_residuals^2))
print(train_rmse)

# testing
test_predictions = predict(fullFit, newdata = test)
testing_residuals = test$PTS - test_predictions
test_rmse = sqrt(mean(testing_residuals^2))
print(test_rmse)
rmse_ratio = test_rmse / train_rmse
print(rmse_ratio)


# lasso
fitLasso = cv.glmnet(x_train, y_train, alpha = 1, nfolds = 7)
plot(fitLasso)

# training
train_lasso_pred = predict(fitLasso, x_train, s = "lambda.1se")
rmse_lasso_train = sqrt(mean((train_lasso_pred - y_train)^2))
print(rmse_lasso_train)

# testing
lassoPred = predict(fitLasso, x_test, s = "lambda.1se")
rmseLasso = sqrt(mean((lassoPred - y_test)^2))
print(rmseLasso)

ssr_lasso = sum((train_lasso_pred - y_train)^2)
tss_lasso = sum((y_train - mean(y_train))^2)
r2_lasso = 1 - (ssr_lasso / tss_lasso)
print(r2_lasso)

lasso_ratio = rmseLasso / rmse_lasso_train
print(lasso_ratio)

selected_vars_lasso = sum(coef(fitLasso, s = "lambda.1se") != 0) - 1
print(selected_vars_lasso)

lasso_coef = coef(fitLasso, s = "lambda.1se")
print(lasso_coef)


# ols
predicted_values = predict(fullFit)
residuals = fullFit$residuals
plot(predicted_values, residuals)

ols_prediction = predict(fullFit, newdata = test)
ols_residuals = test$PTS - ols_prediction
ols_test_rmse = sqrt(mean(ols_residuals^2))
print(ols_test_rmse)

# elastic
# 0.25
fitElastic_25 = cv.glmnet(x_train, y_train, alpha = .25, nfolds = 7)
plot(fitElastic_25)

# training
elastic_25_train = predict(fitElastic_25, x_train, s = "lambda.1se")
rmse_elastic_25_train = sqrt(mean((elastic_25_train - y_train)^2))
print(rmse_elastic_25_train)

# testing
elastic_25_test = predict(fitElastic_25, x_test, s = "lambda.1se")
rmse_elastic_25_test = sqrt(mean((elastic_25_test - y_test)^2))
print(rmse_elastic_25_test)
print(rmse_elastic_25_test / rmse_elastic_25_train)

# 0.50
fitElastic_50 = cv.glmnet(x_train, y_train, alpha = .5, nfolds = 7)
plot(fitElastic_50)

# training
elastic_50_train = predict(fitElastic_50, x_train, s = "lambda.1se")
rmse_elastic_50_train = sqrt(mean((elastic_50_train - y_train)^2))
print(rmse_elastic_50_train)

# testing
elastic_50_test = predict(fitElastic_50, x_test, s = "lambda.1se")
rmse_elastic_50_test = sqrt(mean((elastic_50_test - y_test)^2))
print(rmse_elastic_50_test)
print(rmse_elastic_50_test / rmse_elastic_50_train)

# 0.75
fitElastic_75 = cv.glmnet(x_train, y_train, alpha = .75, nfolds = 7)
plot(fitElastic_75)

# training
elastic_75_train = predict(fitElastic_75, x_train, s = "lambda.1se")
rmse_elastic_75_train = sqrt(mean((elastic_75_train - y_train)^2))
print(rmse_elastic_75_train)

# testing
elastic_75_test = predict(fitElastic_75, x_test, s = "lambda.1se")
rmse_elastic_75_test = sqrt(mean((elastic_75_test - y_test)^2))
print(rmse_elastic_75_test)
print(rmse_elastic_75_test / rmse_elastic_75_train)



# other ways
library(dplyr)
library(kknn)
library(caret)

luka_pca = prcomp(luka)
print(luka_pca)
plot(luka_pca$x)

# decision tree
train_control = trainControl(method = "cv", number = 10)
preproc = c("center", "scale")
luka_tree = train(PTS ~ ., data = train, method = "rpart", trControl = train_control)
print(luka_tree)

# knn
tuneGrid = expand.grid(kmax = 3:7,
                       kernel = c("rectangular", "cos"),
                       distance = 1:3)

luka_knn = train(PTS ~ ., data = train, method = "kknn", trControl = train_control,
                 preProcess = preproc, tuneGrid = tuneGrid)
print(luka_knn)

# plotting
library(ggplot2)
luka_pca_df = as.data.frame(luka_pca$x)
luka_pca_df$tPTS = luka$PTS

luka_pca_df$tree = predict(luka_tree, newdata = luka)
luka_pca_df$knn = predict(luka_knn, newdata = luka)

ggplot(luka_pca_df, aes(x = PC1, y = PC2, color = tree)) +
  geom_point(size = 2) +
  ggtitle("PCA Scatter Plot with Decision Tree")


# clustering
library(car)
library(factoextra)

fviz_nbclust(luka, kmeans, method = "wss")

luka_x = luka[, -14]

preproc = preProcess(luka_x, method = c("center", "scale"))
luka_data = predict(preproc, luka_x)

fit_25 = kmeans(luka_data, centers = 3, nstart = 100)
fviz_cluster(fit_25, data = luka_data)


dist_luka1 = dist(luka_data, method = "euclidean")
hfit1 = hclust(dist_luka1, method = "complete")

dist_luka2 = dist(luka_data, method = "euclidean")
hfit2 = hclust(dist_luka2, method = "average")

dist_luka3 = dist(luka_data, method = "manhattan")
hfit3 = hclust(dist_luka3, method = "complete")

dist_luka4 = dist(luka_data, method = "euclidean")
hfit4 = hclust(dist_luka4, method = "average")

h1 = cutree(hfit1, k=3)
fviz_cluster(list(data = luka_data, cluster = h1))

h2 = cutree(hfit2, k=3)
fviz_cluster(list(data = luka_data, cluster = h2))

h3 = cutree(hfit3, k=3)
fviz_cluster(list(data = luka_data, cluster = h3))

h4 = cutree(hfit4, k=3)
fviz_cluster(list(data = luka_data, cluster = h4))
