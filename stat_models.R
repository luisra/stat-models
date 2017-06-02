# Packages
install.packages("RCurl")
install.packages("leaps")
install.packages("glmnet")
install.packages("MASS")
library(RCurl)
library(leaps)
library(glmnet)
library(MASS)
#

# Get data
x <- getURL("https://raw.githubusercontent.com/luisra/stat-models/master/HR_train.csv")
hr <- read.csv(text = x, header = TRUE, sep = ",")

y <- getURL("https://raw.githubusercontent.com/luisra/stat-models/master/HR_test.csv")
hr.test <- read.csv(text = y, header = TRUE, sep = ",")
#

### Predictive Models ###

# Best Subset Exhaustive 
regfit.full <- regsubsets(satisfaction_level ~ ., hr)
reg.summary <- summary(regfit.full)

reg.summary$cp
reg.summary$bic

plot(regfit.full, scale="Cp")
plot(regfit.full, scale="bic")

cp.model <- lm(satisfaction_level ~ last_evaluation + number_project
               + average_montly_hours + time_spend_company + left + department, hr)

bic.model <- lm(satisfaction_level ~ last_evaluation + number_project
                + average_montly_hours + time_spend_company + left, hr)

cp.predict <- predict(cp.model, newdata = hr.test) 
bic.predict <- predict(bic.model, newdata = hr.test)

cp.MSE <- mean((hr.test$satisfaction_level - cp.predict)^2) 
bic.MSE <- mean((hr.test$satisfaction_level - bic.predict)^2)
#

# Best Subset Forward 
regfit.fwd <- regsubsets(satisfaction_level ~ ., hr, method="forward")
reg.fwd.summary <- summary(regfit.fwd)

reg.fwd.summary$cp
reg.fwd.summary$bic

plot(regfit.fwd, scale="Cp")
plot(regfit.fwd, scale="bic")

cp.fwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                   + average_montly_hours + time_spend_company + left + department, hr)

bic.fwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                    + average_montly_hours + time_spend_company + left, hr)

cp.fwd.predict <- predict(cp.fwd.model, newdata = hr.test) 
bic.fwd.predict <- predict(bic.fwd.model, newdata = hr.test)

cp.fwd.MSE <- mean((hr.test$satisfaction_level - cp.fwd.predict)^2) 
bic.fwd.MSE <- mean((hr.test$satisfaction_level - bic.fwd.predict)^2)
#

# Best Subset Backward 
regfit.bwd <- regsubsets(satisfaction_level ~ ., hr, method="backward")
reg.bwd.summary <- summary(regfit.bwd)

reg.bwd.summary$cp
reg.bwd.summary$bic

plot(regfit.bwd, scale="Cp")
plot(regfit.bwd, scale="bic")

cp.bwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                   + average_montly_hours + time_spend_company + left + department, hr)

bic.bwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                    + average_montly_hours + time_spend_company + left, hr)

cp.bwd.predict <- predict(cp.bwd.model, newdata = hr.test) 
bic.bwd.predict <- predict(bic.bwd.model, newdata = hr.test)

cp.bwd.MSE <- mean((hr.test$satisfaction_level - cp.bwd.predict)^2) 
bic.bwd.MSE <- mean((hr.test$satisfaction_level - bic.bwd.predict)^2)
#

# Ridge 
x <- model.matrix(satisfaction_level ~ ., hr)[,-1]
y <- hr$satisfaction_level
set.seed(4282017)
ridge.cv.out <- cv.glmnet(x, y, alpha=0)
ridge.cv.out$lambda.min

ridge.mod <- glmnet(x, y, alpha=0, lambda =
                      ridge.cv.out$lambda.min)

x_test <- model.matrix(satisfaction_level ~ ., hr.test)[,-1]
ridge.preds <- predict(ridge.mod, newx = x_test)
ridge.MSE <- mean((hr.test$satisfaction_level - ridge.preds)^2)
#

# Lasso 
set.seed(4282017)
lasso.cv.out <- cv.glmnet(x, y, alpha=1)
lasso.cv.out$lambda.min

lasso.mod <- glmnet(x, y, alpha=1, lambda =
                      lasso.cv.out$lambda.min)

lasso.preds <- predict(lasso.mod, newx = x_test)
lasso.MSE <- mean((hr.test$satisfaction_level - lasso.preds)^2)
#

### Classification Models ###

# Logit 
logit.model <- glm(left ~., data = hr, 
                   family = binomial(link=logit))

summary(logit.model)

logit.pred <- predict(logit.model, newdata=hr.test,type="response")
pred<- rep(0,2999) 
pred[logit.pred>0.5]=1
table(pred, hr.test$left)
mean(pred != hr.test$left)
#

# Probit
probit.model <- glm(left ~., data = hr, 
                    family = binomial(link=probit))

summary(probit.model)

probit.pred <- predict(probit.model, newdata=hr.test,type="response")
pred<- rep(0,2999) 
pred[probit.pred>0.5]=1
table(pred, hr.test$left)
mean(pred != hr.test$left)
#

# LDA
lda.fit <- lda(left ~ ., data = hr) 

lda.pred <- predict(lda.fit, hr.test)
lda.class <- lda.pred$class

table(lda.class, hr.test$left)
mean(lda.class != hr.test$left)
#

# QDA
qda.fit <- qda(left ~ ., data = hr) 

qda.pred <- predict(qda.fit, hr.test)
qda.class <- qda.pred$class

table(qda.class, hr.test$left)
mean(qda.class != hr.test$left)
# 