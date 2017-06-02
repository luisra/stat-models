# Predictive and Classification Models for Employee Satisfaction and Retention

Predictive and classification models, based on a dataset including employee information from the Human Resources office of a particular company. The [training](https://raw.githubusercontent.com/luisra/stat-models/master/HR_train.csv) set and [test](https://raw.githubusercontent.com/luisra/stat-models/master/HR_test.csv) set used for this study can be downloaded using the links.

## Synopsis

This project aims to build a predictive model for the satisfaction level of an employee and a classification model for whether or not an employee would leave the company. The study revealed that models built via best subset selection, ridge regression, and lasso are all capable of predicting the satisfaction level of an employee with a MSE of 5%. The study also found the QDA model to be the best overall fit for classifying whether or not an employee would leave the company (12.1% error rate). For whether an employee would stay, the logit model provided better results (7.1% error rate). For whether an employee would leave, the QDA model worked best (17.0% error rate).

## Code Example

The following questions guided our analysis:
* Is it possible to build a predictive model with a mean squared error (MSE) of 5% or less?
* Is it possible to build a classification model with a test error rate of 5% or less?

First, we built and tested predictive models for the satisfaction level of an employee. We picked predictors based on best subset selection, forward stepwise, and backward stepwise. Ridge regression and lasso models were considered as well.

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

Best Subset Selection (Backward): 
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

Next, we proceeded to build and test classification models for whether or not an employee would leave the company. We evaluated both the overall error rate and the class-specific error rate. All predictors were found to be significant for this approach.

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

[WIP]

## Installation

The stat_models.R script performs all aspects of this implementation.

## Tests

[WIP]

Code:
```
I am groot!
```

## License

MIT License
