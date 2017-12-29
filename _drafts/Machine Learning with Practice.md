# Naive Bayes
indepent and assume $p(X_i|y)$ is a gussian or other distribution

<br>

* * *

<br>

# SVM
## Parameters of SVM
- 'C'
    Controls tradeoff between smooth decision boundary and classifying training points correctly

<br>

* * *

<br>

# Decision Tree
The decision tree learns is to use a algorithm to find the decision boundaries automatically based on data.

## Information gain
information gain = entropy(parent) - [weighted average] entropy(chiildren)

## Parameters
- 'min_samples_split': 最小样本分割
    The minimum number of samples that are available the tree to continue to split further.

<br>

* * *

<br>

# k nearest neighbors

<br>

* * *

<br>

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

## Outlier Rejection
One solution for outlier in regression:
1. Train
2. Remove samples with largest residual error(an error after fitting)
3. Re-train

`nan` is ***not a number***

error: actual value - predicted value (target - predict)

<br>

* * *

<br>

# Regression
Minimizing the mean squared error(MSE), MES comes from the sum of squared error(SSE)
actual value $y$
predict value $\hat{y} = wx + b$

Algorithms to solve this problem
- ordinary least squares (OLS)
- gradient descent

<br>

* * *

<br>

# Clustering
## K-Means
K-Means algorithm has four steps:
1. Initialize k cluster center randomly(k is a hyper-paramter)
2. Assign: decide each point belong to which class
3. Optimize: minimize the distance between each point and its class(update cluster center)
4. Loop step.2 and step.3 until it is up the convergence  

For K-means, it is very sensitive to the initialization, and it easy to get the local minimum.

**Note:** K-Means is belong to **hill climbing algorithm**.

<br>

* * *

<br>

# Feature Scaling (Feature Normlizition)
**Key idea:** Scaling different feature into a same scale(often in [0, 1]), in order to balance the influence of different feature with different value range.

**Note:** If there are some outlier in the dataset, it will have a bad influence on the value of maximum and minimum.

# Feature Selection
> Albert Einstein said "Make everything as simple as possible, but no simpler."

- Select best features
- Add new feature

> Features $\ne$ Infomation: what we want is the bare minimum number of features that it takes to give us as much information as possible.

So, the goal of feature selection is to select the features that can give us more information, and delet the features that can't give any inforamtion.

In `sklearn`, you can use:
- `sklearn.feature_selection.SelectKBest`
- `sklearn.feature_selection.SelectPercentile`

# PCA (Principal Component Analysis)
**Key idea:** PCA find a new coordinate system that's obtained from the old one by translation and rotation only. And it moves the center of the coordinate system with the center of the data, moves the x-axis into the principal axis of variation, and moves axis down the road into a orthogonal less important directions of variation.
> Find the direction of maximal variabce, which will minimizes information loss when old data from higher-dimension are projected into lower-dimension.

**Summary:**
- systematized way to transform input features into principal componect
- use principal components as new features
- PCs are directions in data that maximize variance(minimize information loss) when you project/compress down onto them

## When to use PCA
- Latent feature driving the patterns in data
- Dimensionality reduction
    - visualize high-dimensional data
    - reduce noise
    - make other algorithms(regression, classification) work better with fewer input

# Others
## Bias-Variance Dilemma
- high bias
    - the model pays little attention to data
    - oversimplified
    - **high error on training set**
- high variance
    - the model pays too much attention to data(does not generalize well)
    - overfiting
    - **much higher error on test set than on training set**

## Bag of Words
**Key idea:** A bag of words contains a lot of words, then it will inspect each word whether occur in the input sentence or text, and generate a ***dictionary***, which expresses each word's frequency or frequency count.

**Note:** A frequency count (频数) is a measure of the number of times that an event occurs.

In `sklearn`, you can get bag of word from `skearn.feature_extraction`

## TF-IDF
- **TF:** term frequency $\mathrm{t,d}$
    the number of times that term $t$ occurs in document $d$<br>
    **理解：** TF为**词频**，指的是某个给定词语在该文件中出现的频率

    <p>

    $$
    \mathrm{tf}_{i,j} = \frac{n_{i,j}}{\sum_k n_k,j}
    $$

    其中，$n_{i,j}$是该词在文件$d_j$中出现的次数，分母是文件$d_j$中所有词出现的次数之和

    </p>
<br>
    
- **IDF:** inverse document frequency
    **理解：** 是一个词语普遍重要性的度量

    <p>

    $$
    \mathrm{idf}_i = \log\frac{D}{\{j: t_i \in d_j\}}
    $$

    其中，$D$为语料库中的文件总数，$\{j:t_i \in d_j\}$表示包含词语$t_i$的文件数目

    </p>

<br>

可得，

<p>

$$
\mathrm{tfidf}_{ij} = \mathrm{tf}_{i,j} \times \mathrm{idf}_i
$$

</p>

***References:***
- [wiki: ti-idf](https://zh.wikipedia.org/wiki/Tf-idf)


## Preprocess with text
1. Stemmer: use `nltk.stem.snowball`
2. Set stopwords and remove them
3. Calculate TF-IDF
step.2 and step.3 can do together with `sklearn.feature_extraction.text.TfidfVectorizer`
