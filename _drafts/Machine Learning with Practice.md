# Naive Bayes
indepent and assume $p(X_i|y)$ is a gussian or other distribution

# SVM
## Parameters of SVM
- 'C'
    Controls tradeoff between smooth decision boundary and classifying training points correctly

# Decision Tree
The decision tree learns is to use a algorithm to find the decision boundaries automatically based on data.

## Information gain
information gain = entropy(parent) - [weighted average] entropy(chiildren)

## Parameters
- 'min_samples_split': 最小样本分割
    The minimum number of samples that are available the tree to continue to split further.

# k nearest neighbors

# Ensemble methods
## Adaboost

## Random forest

<br>

* * *

<br>

# Dataset and Problem
## Accuracy and Training set size
Here is a truism of machine learning:
> **More Data > Fine-Tuned Algorithm**
Use more data will almost always help out the performance of the algorithm.

## Types of Data
- numerical: numerical values (numbers)
- categorical: limited number of discrete values (category)
- time series: temporal value (data, timestamp)
- text: words
- image


`nan` is ***not a number***

error: actual value - predicted value (target - predict)

# Regression
Minimizing the mean squared error(MSE), MES comes from the sum of squared error(SSE)
actual value $y$
predict value $\hat{y} = wx + b$

Algorithms to solve this problem
- ordinary least squares (OLS)
- gradient descent