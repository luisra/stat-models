# Predictive and Classification Models for Employee Satisfaction and Retention

Predictive and classification models, based on a dataset including employee information from the Human Resources office of a particular company. The [training](https://raw.githubusercontent.com/luisra/stat-models/master/HR_train.csv) set and [test](https://raw.githubusercontent.com/luisra/stat-models/master/HR_test.csv) set used for this study can be downloaded using the links.

## Synopsis

This project aims to build a predictive model for the satisfaction level of an employee and a classification model for whether or not an employee would leave the company. The study revealed that models built via best subset selection, ridge regression, and lasso are all capable of predicting the satisfaction level of an employee with a MSE of 5%. The study also found the QDA model to be the best overall fit for classifying whether or not an employee would leave the company (12.1% error rate). For whether an employee would stay, the logit model provided better results (7.1% error rate). For whether an employee would leave, the QDA model worked best (17.0% error rate).

## Code Example

The following questions guided our analysis:
* Is it possible to build a predictive model with a mean squared error (MSE) of 5% or less?
* Is it possible to build a classification model with a test error rate of 5% or less?

First, we built predictive models for the satisfaction level of an employee. We picked predictors based on best subset selection, forward stepwise, and backward stepwise. Ridge regression and lasso models were considered as well.

Best Subset (Exhaustive): 
```
regfit.full <- regsubsets(satisfaction_level ~ ., hr)

cp.model <- lm(satisfaction_level ~ last_evaluation + number_project
               + average_montly_hours + time_spend_company + left + department, hr)

bic.model <- lm(satisfaction_level ~ last_evaluation + number_project
                + average_montly_hours + time_spend_company + left, hr)
```

Best Subset (Forward):
```
regfit.fwd <- regsubsets(satisfaction_level ~ ., hr, method="forward")

cp.fwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                   + average_montly_hours + time_spend_company + left + department, hr)

bic.fwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                    + average_montly_hours + time_spend_company + left, hr)
```

Best Subset (Backward): 
```
regfit.bwd <- regsubsets(satisfaction_level ~ ., hr, method="backward")

cp.bwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                   + average_montly_hours + time_spend_company + left + department, hr)

bic.bwd.model <- lm(satisfaction_level ~ last_evaluation + number_project
                    + average_montly_hours + time_spend_company + left, hr)
```

Ridge Regression:
```
ridge.mod <- glmnet(x, y, alpha=0, lambda =
                      ridge.cv.out$lambda.min)
```

Lasso:
```
lasso.mod <- glmnet(x, y, alpha=1, lambda =
                      lasso.cv.out$lambda.min)
```

Next, we proceeded to build classification models for whether or not an employee would leave the company. We evaluated both the overall error rate and the class-specific error rate. All predictors were found to be significant for this approach.

Logit:
```
logit.model <- glm(left ~., data = hr, 
                   family = binomial(link=logit))
```

Probit:
```
probit.model <- glm(left ~., data = hr, 
                    family = binomial(link=probit))
```

LDA:
```
lda.fit <- lda(left ~ ., data = hr) 
```

QDA:
```
qda.fit <- qda(left ~ ., data = hr) 
```

## Motivation

We wanted to build the best possible models for predicting the satisfaction level of an employee and classifying whether or not an employee would leave the company. This would allow for greater accuracy in both future predictions and classifications.

## Installation

The stat_models.R script performs all aspects of this implementation.

## Tests

In order to find the best predictive model for the satisfaction level of an employee and the best classification model for whether or not an employee would leave the company, we calculated the MSE of our predictive models and the test error rate of our classification models.

MSE for predictive models --

Best Subset Selection:
```
cp.predict <- predict(cp.model, newdata = hr.test) 
bic.predict <- predict(bic.model, newdata = hr.test)

cp.MSE <- mean((hr.test$satisfaction_level - cp.predict)^2) 
bic.MSE <- mean((hr.test$satisfaction_level - bic.predict)^2)
```

Ridge:
```
ridge.preds <- predict(ridge.mod, newx = x_test)
ridge.MSE <- mean((hr.test$satisfaction_level - ridge.preds)^2)
```

Lasso:
```
lasso.preds <- predict(lasso.mod, newx = x_test)
lasso.MSE <- mean((hr.test$satisfaction_level - lasso.preds)^2)
```

Test error rate for classification models --

Logit:
```
logit.pred <- predict(logit.model, newdata=hr.test,type="response")
pred<- rep(0,2999) 
pred[logit.pred>0.5]=1

table(pred, hr.test$left)
mean(pred != hr.test$left)
```

Probit:
```
probit.pred <- predict(probit.model, newdata=hr.test,type="response")
pred<- rep(0,2999) 
pred[probit.pred>0.5]=1

table(pred, hr.test$left)
mean(pred != hr.test$left)
```

LDA:
```
lda.pred <- predict(lda.fit, hr.test)
lda.class <- lda.pred$class

table(lda.class, hr.test$left)
mean(lda.class != hr.test$left)
```

QDA:
```
qda.pred <- predict(qda.fit, hr.test)
qda.class <- qda.pred$class

table(qda.class, hr.test$left)
mean(qda.class != hr.test$left)
```

## License

MIT License
