# cleaned code
library(ggplot2)
library(dplyr)
library(slider)
library(lubridate)
library(caret)


luka = read.csv("LukaDoncic.csv")
head(luka)
names(luka)[names(luka) == "Unnamed..5"] = "HomeAway"
luka$HomeAway[luka$HomeAway == "@"] = 1
luka$HomeAway = as.integer(luka$HomeAway)
luka$MP = period_to_seconds(ms(luka$MP))
luka[, 7:24] = lapply(luka[, 7:24], as.numeric)
luka = na.omit(luka)

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
# full fit
fullFit = lm(PTS ~ ., data = clean_luka)
summary(fullFit)

pred_full = predict(fullFit, newdata = clean_luka)
data_vis_full = data.frame(Actual = clean_luka$PTS, Predicted = pred_full)
ggplot(data_vis_full, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Full Fit Actual vs. Predicted PTS", x = "Actual PTS", y = "Predicted PTS")

full = lm(PTS ~ ., data = clean_luka)
null = lm(PTS ~ 1, data = clean_luka)

# stepwise
luka_step = step(null, scope = list(lower=null, upper=full), direction = "both", trace = F)
summary(luka_step)
model_step = lm(PTS ~ GMS_avg10 + TOV_career_sd + FT_avg3 + TOV_avg10 + 
                  TOV_sd5 + AST_sd3 + STL_sd3 + FG._avg3 + X3PA_sd10 + STL_avg10 + 
                  STL_avg3 + BLK_avg5 + AST_avg5 + TOV_career_avg + FG._avg10 + 
                  MP_avg3 + PM_sd3 + FT_sd10 + HomeAway + RB_sd10 + PF_sd10 + 
                  X3P._sd5 + PF_sd5, data = clean_luka)

predicted_step = predict(model_step)
data_vis_step = data.frame(Actual = clean_luka$PTS, Predicted = predicted_step)
ggplot(data_vis_step, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Stepwise Actual vs. Predicted PTS", x = "Actual PTS", y = "Predicted PTS")


# forward
luka_forward = step(null, scope = list(lower=null, upper=full), direction = "forward", trace = F)
summary(luka_forward)
model_forward = lm(PTS ~ GMS_avg10 + TOV_career_sd + FT_avg3 + TOV_avg10 + 
                     TOV_sd5 + AST_sd3 + STL_sd3 + FG._avg3 + X3PA_sd10 + STL_avg10 + 
                     STL_avg3 + BLK_avg5 + RB_sd3 + AST_avg5 + TOV_career_avg + 
                     FG._avg10 + MP_avg3 + PM_sd3 + FT_sd10 + HomeAway + RB_sd10 + 
                     PF_sd10 + X3P._sd5 + PF_sd5, data = clean_luka)
predicted_forward = predict(model_forward)
data_vis_forward = data.frame(Actual = clean_luka$PTS, Predicted = predicted_forward)
ggplot(data_vis_forward, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Forward Actual vs. Predicted PTS", x = "Actual PTS", y = "Predicted PTS")

# backward
luka_backward = step(full, direction = "backward", trace = F)
summary(luka_backward)
model_backward = lm(PTS ~ HomeAway + PTS_avg5 + PTS_sd10 + PTS_career_sd + 
                      FG_avg3 + FG_avg5 + FG_career_avg + FGA_avg5 + FGA_career_avg + 
                      FG._sd3 + FG._sd5 + X3P_career_avg + X3PA_avg3 + X3PA_sd10 + 
                      X3PA_career_avg + X3P._sd10 + X3P._career_avg + FT_career_sd + 
                      FTA_avg3 + FT._sd3 + FT._avg10 + RB_sd10 + AST_sd3 + AST_sd10 + 
                      AST_career_sd + STL_avg3 + STL_sd3 + STL_avg10 + BLK_avg5 + 
                      BLK_sd5 + BLK_avg10 + BLK_career_avg + BLK_career_sd + TOV_avg5 + 
                      TOV_sd5 + TOV_avg10 + TOV_sd10 + TOV_career_avg + PF_career_avg + 
                      GMS_avg5 + GMS_sd10 + GMS_career_avg + GMS_career_sd + PM_sd3 + 
                      PM_avg10 + PM_career_avg + MP_sd3 + MP_avg5 + MP_sd5 + MP_career_avg, 
                    data = clean_luka)
predicted_backward = predict(model_backward)
data_vis_backward = data.frame(Actual = clean_luka$PTS, Predicted = predicted_backward)
ggplot(data_vis_backward, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Backward Actual vs. Predicted PTS", x = "Actual PTS", y = "Predicted PTS")



# testing
# full fit
prediction_full = predict(fullFit, newdata = test)
mse = mean((prediction_full - test$PTS)^2)
rmse = sqrt(mse)
mae = mean(abs(prediction_full - test$PTS))
cat("RMSE:", rmse, "\nMAE:", mae, "\n")

# stepwise
prediction_step = predict(model_step, newdata = test)
mse = mean((prediction_step - test$PTS)^2)
rmse = sqrt(mse)
mae = mean(abs(prediction_step - test$PTS))
cat("RMSE:", rmse, "\nMAE:", mae, "\n")

# forward
prediction_forward = predict(model_forward, newdata = test)
mse = mean((prediction_forward - test$PTS)^2)
rmse = sqrt(mse)
mae = mean(abs(prediction_forward - test$PTS))
cat("RMSE:", rmse, "\nMAE:", mae, "\n")

# backward
prediction_backward = predict(model_backward, newdata = test)
mse = mean((prediction_backward - test$PTS)^2)
rmse = sqrt(mse)
mae = mean(abs(prediction_backward - test$PTS))
cat("RMSE:", rmse, "\nMAE:", mae, "\n")





# cross validation
train_control = trainControl(method = "cv", number = 10)

# full fit
cv_full = train(PTS ~ ., 
                 data = clean_luka, 
                 method = "lm", 
                 trControl = train_control)
print(cv_full)

# stepwise
formula_step = PTS ~ GMS_avg10 + TOV_career_sd + FT_avg3 + TOV_avg10 +
  TOV_sd5 + AST_sd3 + STL_sd3 + FG._avg3 + X3PA_sd10 + STL_avg10 +
  STL_avg3 + BLK_avg5 + AST_avg5 + TOV_career_avg + FG._avg10 +
  MP_avg3 + PM_sd3 + FT_sd10 + HomeAway + RB_sd10 + PF_sd10 +
  X3P._sd5 + PF_sd5
cv_step = train(formula_step, 
                 data = clean_luka, 
                 method = "lm", 
                 trControl = train_control)
print(cv_step)

# forward
formula_forward = PTS ~ GMS_avg10 + TOV_career_sd + FT_avg3 + TOV_avg10 + 
  TOV_sd5 + AST_sd3 + STL_sd3 + FG._avg3 + X3PA_sd10 + STL_avg10 + 
  STL_avg3 + BLK_avg5 + RB_sd3 + AST_avg5 + TOV_career_avg + 
  FG._avg10 + MP_avg3 + PM_sd3 + FT_sd10 + HomeAway + RB_sd10 + 
  PF_sd10 + X3P._sd5 + PF_sd5
cv_forward = train(formula_forward,
                   data = clean_luka,
                   method = "lm",
                   trControl = train_control)
print(cv_forward)

# backward
formula_backward = PTS ~ HomeAway + PTS_avg5 + PTS_sd10 + PTS_career_sd + 
  FG_avg3 + FG_avg5 + FG_career_avg + FGA_avg5 + FGA_career_avg + 
  FG._sd3 + FG._sd5 + X3P_career_avg + X3PA_avg3 + X3PA_sd10 + 
  X3PA_career_avg + X3P._sd10 + X3P._career_avg + FT_career_sd + 
  FTA_avg3 + FT._sd3 + FT._avg10 + RB_sd10 + AST_sd3 + AST_sd10 + 
  AST_career_sd + STL_avg3 + STL_sd3 + STL_avg10 + BLK_avg5 + 
  BLK_sd5 + BLK_avg10 + BLK_career_avg + BLK_career_sd + TOV_avg5 + 
  TOV_sd5 + TOV_avg10 + TOV_sd10 + TOV_career_avg + PF_career_avg + 
  GMS_avg5 + GMS_sd10 + GMS_career_avg + GMS_career_sd + PM_sd3 + 
  PM_avg10 + PM_career_avg + MP_sd3 + MP_avg5 + MP_sd5 + MP_career_avg
cv_backward = train(formula_backward,
                    data = clean_luka,
                    method = "lm",
                    trControl = train_control)
print(cv_backward)
