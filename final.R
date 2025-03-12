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


# with every nba player
everyone = read.csv("all_seasons.csv")
everyone = everyone[, -c(1:4, 7:9, 22)]
everyone$draft_round = as.numeric(everyone$draft_round)
everyone$draft_number = as.numeric(as.character(everyone$draft_number))
everyone = na.omit(everyone)
fviz_nbclust(everyone, kmeans, method = "wss")

preproc = preProcess(everyone, method = c("center", "scale"))
ds = predict(preproc, everyone)

fit_25 = kmeans(ds, centers = 5, nstart = 25)
fviz_cluster(fit_25, data = everyone)

head(everyone)













# trying again
luka = read.csv("LukaDoncic.csv")
head(luka)
names(luka)[names(luka) == "Unnamed..5"] = "HomeAway"
luka$HomeAway[luka$HomeAway == "@"] = 1
luka$HomeAway = as.integer(luka$HomeAway)
luka$MP = period_to_seconds(ms(luka$MP))
luka[, 7:24] = lapply(luka[, 7:24], as.numeric)
luka = na.omit(luka)

library(slider)
library(dplyr)
luka = luka %>% mutate(
  PTS_avg3 = lag(slide_dbl(PTS, mean, .before = 3, .after = -1)),
  PTS_sd3 = lag(slide_dbl(PTS, sd, .before = 3, .after = -1)),
  PTS_avg5 = lag(slide_dbl(PTS, mean, .before = 5, .after = -1)),
  PTS_sd5 = lag(slide_dbl(PTS, sd, .before = 5, .after = -1)),
  PTS_avg10 = lag(slide_dbl(PTS, mean, .before = 10, .after = -1)),
  PTS_sd10 = lag(slide_dbl(PTS, sd, .before = 10, .after = -1)),
  PTS_career_avg = lag(slide_dbl(PTS, mean, .before = 433, .after = -1)),
  PTS_career_sd = lag(slide_dbl(PTS, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  FG_avg3 = lag(slide_dbl(FG, mean, .before = 3, .after = -1)),
  FG_sd3 = lag(slide_dbl(FG, sd, .before = 3, .after = -1)),
  FG_avg5 = lag(slide_dbl(FG, mean, .before = 5, .after = -1)),
  FG_sd5 = lag(slide_dbl(FG, sd, .before = 5, .after = -1)),
  FG_avg10 = lag(slide_dbl(FG, mean, .before = 10, .after = -1)),
  FG_sd10 = lag(slide_dbl(FG, sd, .before = 10, .after = -1)),
  FG_career_avg = lag(slide_dbl(FG, mean, .before = 433, .after = -1)),
  FG_career_sd = lag(slide_dbl(FG, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  FGA_avg3 = lag(slide_dbl(FGA, mean, .before = 3, .after = -1)),
  FGA_sd3 = lag(slide_dbl(FGA, sd, .before = 3, .after = -1)),
  FGA_avg5 = lag(slide_dbl(FGA, mean, .before = 5, .after = -1)),
  FGA_sd5 = lag(slide_dbl(FGA, sd, .before = 5, .after = -1)),
  FGA_avg10 = lag(slide_dbl(FGA, mean, .before = 10, .after = -1)),
  FGA_sd10 = lag(slide_dbl(FGA, sd, .before = 10, .after = -1)),
  FGA_career_avg = lag(slide_dbl(FGA, mean, .before = 433, .after = -1)),
  FGA_career_sd = lag(slide_dbl(FGA, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  FG._avg3 = lag(slide_dbl(FG., mean, .before = 3, .after = -1)),
  FG._sd3 = lag(slide_dbl(FG., sd, .before = 3, .after = -1)),
  FG._avg5 = lag(slide_dbl(FG., mean, .before = 5, .after = -1)),
  FG._sd5 = lag(slide_dbl(FG., sd, .before = 5, .after = -1)),
  FG._avg10 = lag(slide_dbl(FG., mean, .before = 10, .after = -1)),
  FG._sd10 = lag(slide_dbl(FG., sd, .before = 10, .after = -1)),
  FG._career_avg = lag(slide_dbl(FG., mean, .before = 433, .after = -1)),
  FG._career_sd = lag(slide_dbl(FG., sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  X3P_avg3 = lag(slide_dbl(X3P, mean, .before = 3, .after = -1)),
  X3P_sd3 = lag(slide_dbl(X3P, sd, .before = 3, .after = -1)),
  X3P_avg5 = lag(slide_dbl(X3P, mean, .before = 5, .after = -1)),
  X3P_sd5 = lag(slide_dbl(X3P, sd, .before = 5, .after = -1)),
  X3P_avg10 = lag(slide_dbl(X3P, mean, .before = 10, .after = -1)),
  X3P_sd10 = lag(slide_dbl(X3P, sd, .before = 10, .after = -1)),
  X3P_career_avg = lag(slide_dbl(X3P, mean, .before = 433, .after = -1)),
  X3P_career_sd = lag(slide_dbl(X3P, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  X3PA_avg3 = lag(slide_dbl(X3PA, mean, .before = 3, .after = -1)),
  X3PA_sd3 = lag(slide_dbl(X3PA, sd, .before = 3, .after = -1)),
  X3PA_avg5 = lag(slide_dbl(X3PA, mean, .before = 5, .after = -1)),
  X3PA_sd5 = lag(slide_dbl(X3PA, sd, .before = 5, .after = -1)),
  X3PA_avg10 = lag(slide_dbl(X3PA, mean, .before = 10, .after = -1)),
  X3PA_sd10 = lag(slide_dbl(X3PA, sd, .before = 10, .after = -1)),
  X3PA_career_avg = lag(slide_dbl(X3PA, mean, .before = 433, .after = -1)),
  X3PA_career_sd = lag(slide_dbl(X3PA, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  X3P._avg3 = lag(slide_dbl(X3P., mean, .before = 3, .after = -1)),
  X3P._sd3 = lag(slide_dbl(X3P., sd, .before = 3, .after = -1)),
  X3P._avg5 = lag(slide_dbl(X3P., mean, .before = 5, .after = -1)),
  X3P._sd5 = lag(slide_dbl(X3P., sd, .before = 5, .after = -1)),
  X3P._avg10 = lag(slide_dbl(X3P., mean, .before = 10, .after = -1)),
  X3P._sd10 = lag(slide_dbl(X3P., sd, .before = 10, .after = -1)),
  X3P._career_avg = lag(slide_dbl(X3P., mean, .before = 433, .after = -1)),
  X3P._career_sd = lag(slide_dbl(X3P., sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  FT_avg3 = lag(slide_dbl(FT, mean, .before = 3, .after = -1)),
  FT_sd3 = lag(slide_dbl(FT, sd, .before = 3, .after = -1)),
  FT_avg5 = lag(slide_dbl(FT, mean, .before = 5, .after = -1)),
  FT_sd5 = lag(slide_dbl(FT, sd, .before = 5, .after = -1)),
  FT_avg10 = lag(slide_dbl(FT, mean, .before = 10, .after = -1)),
  FT_sd10 = lag(slide_dbl(FT, sd, .before = 10, .after = -1)),
  FT_career_avg = lag(slide_dbl(FT, mean, .before = 433, .after = -1)),
  FT_career_sd = lag(slide_dbl(FT, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  FTA_avg3 = lag(slide_dbl(FTA, mean, .before = 3, .after = -1)),
  FTA_sd3 = lag(slide_dbl(FTA, sd, .before = 3, .after = -1)),
  FTA_avg5 = lag(slide_dbl(FTA, mean, .before = 5, .after = -1)),
  FTA_sd5 = lag(slide_dbl(FTA, sd, .before = 5, .after = -1)),
  FTA_avg10 = lag(slide_dbl(FTA, mean, .before = 10, .after = -1)),
  FTA_sd10 = lag(slide_dbl(FTA, sd, .before = 10, .after = -1)),
  FTA_career_avg = lag(slide_dbl(FTA, mean, .before = 433, .after = -1)),
  FTA_career_sd = lag(slide_dbl(FTA, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  FT._avg3 = lag(slide_dbl(FT., mean, .before = 3, .after = -1)),
  FT._sd3 = lag(slide_dbl(FT., sd, .before = 3, .after = -1)),
  FT._avg5 = lag(slide_dbl(FT., mean, .before = 5, .after = -1)),
  FT._sd5 = lag(slide_dbl(FT., sd, .before = 5, .after = -1)),
  FT._avg10 = lag(slide_dbl(FT., mean, .before = 10, .after = -1)),
  FT._sd10 = lag(slide_dbl(FT., sd, .before = 10, .after = -1)),
  FT._career_avg = lag(slide_dbl(FT., mean, .before = 433, .after = -1)),
  FT._career_sd = lag(slide_dbl(FT., sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  RB_avg3 = lag(slide_dbl(TRB, mean, .before = 3, .after = -1)),
  RB_sd3 = lag(slide_dbl(TRB, sd, .before = 3, .after = -1)),
  RB_avg5 = lag(slide_dbl(TRB, mean, .before = 5, .after = -1)),
  RB_sd5 = lag(slide_dbl(TRB, sd, .before = 5, .after = -1)),
  RB_avg10 = lag(slide_dbl(TRB, mean, .before = 10, .after = -1)),
  RB_sd10 = lag(slide_dbl(TRB, sd, .before = 10, .after = -1)),
  RB_career_avg = lag(slide_dbl(TRB, mean, .before = 433, .after = -1)),
  RB_career_sd = lag(slide_dbl(TRB, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  AST_avg3 = lag(slide_dbl(AST, mean, .before = 3, .after = -1)),
  AST_sd3 = lag(slide_dbl(AST, sd, .before = 3, .after = -1)),
  AST_avg5 = lag(slide_dbl(AST, mean, .before = 5, .after = -1)),
  AST_sd5 = lag(slide_dbl(AST, sd, .before = 5, .after = -1)),
  AST_avg10 = lag(slide_dbl(AST, mean, .before = 10, .after = -1)),
  AST_sd10 = lag(slide_dbl(AST, sd, .before = 10, .after = -1)),
  AST_career_avg = lag(slide_dbl(AST, mean, .before = 433, .after = -1)),
  AST_career_sd = lag(slide_dbl(AST, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  STL_avg3 = lag(slide_dbl(STL, mean, .before = 3, .after = -1)),
  STL_sd3 = lag(slide_dbl(STL, sd, .before = 3, .after = -1)),
  STL_avg5 = lag(slide_dbl(STL, mean, .before = 5, .after = -1)),
  STL_sd5 = lag(slide_dbl(STL, sd, .before = 5, .after = -1)),
  STL_avg10 = lag(slide_dbl(STL, mean, .before = 10, .after = -1)),
  STL_sd10 = lag(slide_dbl(STL, sd, .before = 10, .after = -1)),
  STL_career_avg = lag(slide_dbl(STL, mean, .before = 433, .after = -1)),
  STL_career_sd = lag(slide_dbl(STL, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  BLK_avg3 = lag(slide_dbl(BLK, mean, .before = 3, .after = -1)),
  BLK_sd3 = lag(slide_dbl(BLK, sd, .before = 3, .after = -1)),
  BLK_avg5 = lag(slide_dbl(BLK, mean, .before = 5, .after = -1)),
  BLK_sd5 = lag(slide_dbl(BLK, sd, .before = 5, .after = -1)),
  BLK_avg10 = lag(slide_dbl(BLK, mean, .before = 10, .after = -1)),
  BLK_sd10 = lag(slide_dbl(BLK, sd, .before = 10, .after = -1)),
  BLK_career_avg = lag(slide_dbl(BLK, mean, .before = 433, .after = -1)),
  BLK_career_sd = lag(slide_dbl(BLK, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  TOV_avg3 = lag(slide_dbl(TOV, mean, .before = 3, .after = -1)),
  TOV_sd3 = lag(slide_dbl(TOV, sd, .before = 3, .after = -1)),
  TOV_avg5 = lag(slide_dbl(TOV, mean, .before = 5, .after = -1)),
  TOV_sd5 = lag(slide_dbl(TOV, sd, .before = 5, .after = -1)),
  TOV_avg10 = lag(slide_dbl(TOV, mean, .before = 10, .after = -1)),
  TOV_sd10 = lag(slide_dbl(TOV, sd, .before = 10, .after = -1)),
  TOV_career_avg = lag(slide_dbl(TOV, mean, .before = 433, .after = -1)),
  TOV_career_sd = lag(slide_dbl(TOV, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  PF_avg3 = lag(slide_dbl(PF, mean, .before = 3, .after = -1)),
  PF_sd3 = lag(slide_dbl(PF, sd, .before = 3, .after = -1)),
  PF_avg5 = lag(slide_dbl(PF, mean, .before = 5, .after = -1)),
  PF_sd5 = lag(slide_dbl(PF, sd, .before = 5, .after = -1)),
  PF_avg10 = lag(slide_dbl(PF, mean, .before = 10, .after = -1)),
  PF_sd10 = lag(slide_dbl(PF, sd, .before = 10, .after = -1)),
  PF_career_avg = lag(slide_dbl(PF, mean, .before = 433, .after = -1)),
  PF_career_sd = lag(slide_dbl(PF, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  GMS_avg3 = lag(slide_dbl(GmSc, mean, .before = 3, .after = -1)),
  GMS_sd3 = lag(slide_dbl(GmSc, sd, .before = 3, .after = -1)),
  GMS_avg5 = lag(slide_dbl(GmSc, mean, .before = 5, .after = -1)),
  GMS_sd5 = lag(slide_dbl(GmSc, sd, .before = 5, .after = -1)),
  GMS_avg10 = lag(slide_dbl(GmSc, mean, .before = 10, .after = -1)),
  GMS_sd10 = lag(slide_dbl(GmSc, sd, .before = 10, .after = -1)),
  GMS_career_avg = lag(slide_dbl(GmSc, mean, .before = 433, .after = -1)),
  GMS_career_sd = lag(slide_dbl(GmSc, sd, .before = 433, .after = -1)),
)

luka = luka %>% mutate(
  PM_avg3 = lag(slide_dbl(X..., mean, .before = 3, .after = -1)),
  PM_sd3 = lag(slide_dbl(X..., sd, .before = 3, .after = -1)),
  PM_avg5 = lag(slide_dbl(X..., mean, .before = 5, .after = -1)),
  PM_sd5 = lag(slide_dbl(X..., sd, .before = 5, .after = -1)),
  PM_avg10 = lag(slide_dbl(X..., mean, .before = 10, .after = -1)),
  PM_sd10 = lag(slide_dbl(X..., sd, .before = 10, .after = -1)),
  PM_career_avg = lag(slide_dbl(X..., mean, .before = 433, .after = -1)),
  PM_career_sd = lag(slide_dbl(X..., sd, .before = 433, .after = -1)),
)


luka = luka %>% mutate(
  MP_avg3 = lag(slide_dbl(MP, mean, .before = 3, .after = -1)),
  MP_sd3 = lag(slide_dbl(MP, sd, .before = 3, .after = -1)),
  MP_avg5 = lag(slide_dbl(MP, mean, .before = 5, .after = -1)),
  MP_sd5 = lag(slide_dbl(MP, sd, .before = 5, .after = -1)),
  MP_avg10 = lag(slide_dbl(MP, mean, .before = 10, .after = -1)),
  MP_sd10 = lag(slide_dbl(MP, sd, .before = 10, .after = -1)),
  MP_career_avg = lag(slide_dbl(MP, mean, .before = 433, .after = -1)),
  MP_career_sd = lag(slide_dbl(MP, sd, .before = 433, .after = -1)),
)

head(luka)
clean_luka = luka[, -c(1:2, 5:21, 23:24)]
clean_luka = na.omit(clean_luka)
write.csv(clean_luka, "LukaDoncicCleaned.csv")

set.seed(123)
sample = sample(c(TRUE, FALSE), nrow(clean_luka), replace = TRUE, prob = c(0.8, 0.2))
train = clean_luka[sample, ]
test = clean_luka[!sample, ]

# splitting
x_train = as.matrix(train[, -which(colnames(train) == "PTS")])
y_train = train$PTS
x_test = as.matrix(test[, -which(colnames(test) == "PTS")])
y_test = test$PTS

# linear regression
fullFit = lm(PTS ~ ., data = clean_luka)
summary(fullFit)
library(car)
vif(fullFit)

pred_full = predict(fullFit)


full = lm(PTS ~ ., data = clean_luka)
null = lm(PTS ~ 1, data = clean_luka)

# stepwise
luka_step = step(null, scope = list(lower=null, upper=full), direction = "both", trace = F)
summary(luka_step)
# Fit your model
model_step <- lm(PTS ~ GMS_avg10 + TOV_career_sd + FT_avg3 + TOV_avg10 + 
                   TOV_sd5 + AST_sd3 + STL_sd3 + FG._avg3 + X3PA_sd10 + STL_avg10 + 
                   STL_avg3 + BLK_avg5 + AST_avg5 + TOV_career_avg + FG._avg10 + 
                   MP_avg3 + PM_sd3 + FT_sd10 + HomeAway + RB_sd10 + PF_sd10 + 
                   X3P._sd5 + PF_sd5, data = clean_luka)

# Set up a 2x2 plotting area and generate diagnostic plots
par(mfrow = c(1, 1))
plot(model)

# Get predicted values
predicted_step <- predict(model_step)
actual_step <- clean_luka$PTS

# Plot Actual vs Predicted
plot(actual_step, predicted_step, 
     xlab = "Actual PTS", 
     ylab = "Predicted PTS", 
     main = "Actual vs. Predicted PTS")
abline(0, 1, col = 1, "red", lwd = 2)

# Plot using ggplot2
ggplot(data_vis, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Actual vs. Predicted PTS", x = "Actual PTS", y = "Predicted PTS")
data_vis <- data.frame(Actual = clean_luka$PTS, Predicted = predict(model_step))





luka_forward = step(null, scope = list(lower=null, upper=full), direction = "forward", trace = F)
summary(luka_forward)
model_forward = lm(PTS ~ GMS_avg10 + TOV_career_sd + FT_avg3 + TOV_avg10 + 
                     TOV_sd5 + AST_sd3 + STL_sd3 + FG._avg3 + X3PA_sd10 + STL_avg10 + 
                     STL_avg3 + BLK_avg5 + RB_sd3 + AST_avg5 + TOV_career_avg + 
                     FG._avg10 + MP_avg3 + PM_sd3 + FT_sd10 + HomeAway + RB_sd10 + 
                     PF_sd10 + X3P._sd5 + PF_sd5, data = clean_luka)
actual_forward = clean_luka$PTS
data_vis_forward = data.frame(Actual = clean_luka$PTS, Predicted = predict(model_forward))
ggplot(data_vis_forward, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Actual vs. Predicted PTS", x = "Actual PTS", y = "Predicted PTS")

library(ggplot2)

# Create a data frame with actual and predicted values
data_vis <- data.frame(Actual = clean_luka$PTS, Predicted = predict(model_step))



luka_backward = step(full, direction = "backward", trace = F)
summary(luka_backward)
model_backward = lm(formula = PTS ~ HomeAway + Season + PTS_avg3 + PTS_sd3 + PTS_avg5 + 
                      PTS_sd10 + FG_avg3 + FG_avg5 + FG_sd10 + FG_career_avg + 
                      FG_career_sd + FGA_sd5 + FGA_sd10 + FGA_career_avg + FGA_career_sd + 
                      FG._avg5 + FG._career_avg + FG._career_sd + X3P_avg3 + X3P_career_avg + 
                      X3P_career_sd + X3PA_avg3 + X3PA_sd5 + X3PA_avg10 + X3PA_sd10 + 
                      X3PA_career_avg + X3P._career_avg + X3P._career_sd + FT_career_sd + 
                      FTA_sd10 + FTA_career_sd + FT._avg3 + FT._sd3 + RB_sd3 + 
                      RB_sd10 + RB_career_sd + AST_sd10 + AST_career_sd + STL_sd3 + 
                      STL_career_sd + BLK_avg3 + BLK_sd3 + BLK_avg5 + BLK_sd5 + 
                      BLK_career_sd + TOV_avg3 + TOV_avg5 + TOV_sd5 + TOV_avg10 + 
                      TOV_sd10 + TOV_career_avg + PF_sd5 + PF_sd10 + PF_career_avg + 
                      PF_career_sd + GMS_avg5 + GMS_sd10 + GMS_career_avg + GMS_career_sd + 
                      PM_sd5 + PM_career_sd + MP_sd3 + MP_sd10 + MP_career_avg, 
                    data = clean_luka)
predicted_backward = predict(model_backward)
actual = clean_luka$PTS
data_vis_backward = data.frame(Actual = clean_luka$PTS, Predicted = predict(model_backward))
ggplot(data_vis_backward, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Actual vs. Predicted PTS", x = "Actual PTS", y = "Predicted PTS")




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



#lasso
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


# ols
predicted_values = predict(fullFit)
residuals = fullFit$residuals
plot(predicted_values, residuals)

ols_prediction = predict(fullFit, newdata = test)
ols_residuals = test$PTS - ols_prediction
ols_test_rmse = sqrt(mean(ols_residuals^2))
print(ols_test_rmse)


library(pls)
pcr_fit <- pcr(PTS ~ ., data = clean_luka, scale = TRUE, validation = "CV")
summary(pcr_fit)









library(dplyr)
library(slider)
library(lubridate)
library(caret)

# 1. Read and clean the data
luka <- read.csv("LukaDoncic.csv", stringsAsFactors = FALSE)

# Remove rows where player did not play (adjust column names as needed)
luka <- luka %>% filter(!GS %in% c("Inactive", "Did Not Dress"))

# Convert Date and sort chronologically
luka <- luka %>%
  mutate(Date = as.Date(Date),
         Rest = as.numeric(difftime(Date, lag(Date), units = "days")),
         Rest = ifelse(is.na(Rest), 0, Rest),
         DayOfWeek = wday(Date)) %>%
  arrange(Date)

# 2. Feature Engineering: Compute rolling averages and standard deviations
# Example: for Points (PTS)
for(window in c(3, 5, 10)) {
  luka <- luka %>%
    mutate(!!paste0("PTS_last_", window, "_avg") := lag(slide_dbl(PTS, mean, .before = window - 1, .after = 0)),
           !!paste0("PTS_last_", window, "_sd")  := lag(slide_dbl(PTS, sd,   .before = window - 1, .after = 0)))
}

# You can do the same for other stats like FG%, 3P%, FT%, AST, TRB, TOV, etc.
# For example, if you have shooting percentage columns named "FG.", "3P%", "FT%" you can repeat the above loop.

# 3. Select relevant features and remove rows with NA (first few rows might be NA due to rolling calculations)
model_data <- luka %>%
  select(PTS, Rest, DayOfWeek, starts_with("PTS_last_")) %>%
  na.omit()

# 4. Split into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(model_data$PTS, p = 0.7, list = FALSE)
train_data <- model_data[trainIndex, ]
test_data  <- model_data[-trainIndex, ]

# 5. Scale features (except the target 'PTS')
scaler <- preProcess(train_data[, -1], method = c("center", "scale"))
train_scaled <- predict(scaler, train_data[, -1])
test_scaled  <- predict(scaler, test_data[, -1])

# 6. Train a Linear Regression Model using caret with cross-validation
train_control <- trainControl(method = "cv", number = 5)
lm_model <- train(PTS ~ ., data = cbind(PTS = train_data$PTS, train_scaled),
                  method = "lm",
                  trControl = train_control)

print(lm_model)

# 7. Make predictions on the test set
predictions <- predict(lm_model, newdata = test_scaled)

# Evaluate RMSE
rmse <- sqrt(mean((predictions - test_data$PTS)^2))
cat("Test RMSE:", rmse, "\n")

# 8. (Optional) Get prediction intervals from an lm object
# You can also fit an lm model directly to get prediction intervals:
lm_fit <- lm(PTS ~ ., data = cbind(PTS = train_data$PTS, train_scaled))
pred_interval <- predict(lm_fit, newdata = test_scaled, interval = "prediction", level = 0.95)
head(pred_interval)



lm_fit <- lm(PTS ~ ., data = cbind(PTS = train_data$PTS, train_scaled))

# Calculate Cook's distance
cooksd <- cooks.distance(lm_fit)

# Identify influential points (a common rule is Cook's distance > 4/n)
influential_points <- which(cooksd > (4 / nrow(train_data)))

# Remove influential observations
train_data_clean <- train_data[-influential_points, ]
train_scaled_clean <- train_scaled[-influential_points, ]

# Refit the model on the cleaned data
lm_fit_clean <- lm(PTS ~ ., data = cbind(PTS = train_data_clean$PTS, train_scaled_clean))
summary(lm_fit_clean)


