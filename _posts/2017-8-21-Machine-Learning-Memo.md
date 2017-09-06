# 生成式模型和判别式模型
如果从概率的角度理解，特征变量用$x$表示，标签用$y$表示，<br>
**生成式模型：** 对样本的特征和标签的$p(x|y), p(y)$进行建模，之后根据贝叶斯定理$p(x, y) = p(x|y)p(y)$得到联合概率<br>
**判别式模型：** 对样本的特征和标签的$p(y|x)$进行建模

# Gaussian Discriminant Analysis(GDA)
**高斯分布：** $n$维的高斯分布的参数为均指向量(mean vector) $\mathbf{\mu} \in \mathbb{R}^n$和协方差矩阵(covariance matrix)$\mathbf{\Sigma} \in \mathbb{R}^{n \times n}$，记做$x \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$,
<p>

$$
p(\mathbf{x}; \mu, \mathbf{\Sigma}) = \frac{1}{(2\pi)^{n/2}|\mathbf{\Sigma}|^{1/2}}\exp(-\frac{1}{2}(x - \mu)^T\mathbf{\Sigma}^{-1}(\mathbf{x} - \mu))
$$

</p>

**GDA** 是一种生成式学习方法。其假设$p(x|y)$是服从高斯分布的。以二分类为例，
<p>

$$
y \sim Bernoulli(\phi)
$$
$$
\mathbf{x}|y=0 \sim \mathcal{N}(\mu_0, \mathbf{\Sigma})
$$
$$
\mathbf{x}|y=1 \sim \mathcal{N}(\mu_1, \mathbf{\Sigma})
$$

</p>

<!-- 注：假设的$p(\mathbf{x}|y)$的分布，其均值向量是不同的，但是协方差矩阵是相同的，因为在不同的$y$下，变量的均值可以发生改变，但是其特征与特征之间的关系，我们一般假设是不会发生改变的。<br> -->

**GDA和Logistic：** 如果$\mathbf{x}|y=y_i \sim \mathcal{N}(\mu, \sigma^2)$，则后验概率$p(y|\mathbf{x}; \phi, \Sigma, \mu_0, \mu_1) = \frac{1}{1 + \exp(-\theta^Tx)}$，是一个logistic函数。因此当我们确定或比较确定$p(\mathbf{x}|y)$是服从高斯分布时（近似服从高斯），我们使用GDA的效果会比使用logistic要好，因为我们更好的利用了数据的信息。反之，如果我们不确定其实服从高斯分布的，选择logisitic会好一些。<br>

更一般的，如果$\mathbf{x}|y=y_i$ exponential distribution分布，其后验概率$p(y|\mathbf{x})$也是满足logistic。

## 朴素贝叶斯（Naive Bayse）
**核心思想：**
假设在给定$y$的情况下，随机向量$\mathbf{x} = (x_1, x_2, ..., x_n)$的每个变量$x_i$是条件独立的，即，
<p>

$$
p(x_1, x_2, ..., x_n|y) = p(x_1|y)p(x_2|x_1,y)...p(x_n|x_1, x_2,...,x_{n-1}, y) = \prod_{i=1}^{n}p(x_i|y)
$$

</p>

***补充：***
- 概率论的两个核心公式
    - sum rule
        <p>

        $$
        p(A) = \sum_B p(A, B)
        $$

        </p>
    - product rule
        <p>

        $$
        p(A, B) = p(A|B)p(B) = p(B|A)p(A)
        $$

        </p>

- 贝叶斯定理
    <p>

    $$
    p(A|B) = \frac{p(B|A)p(A)}{p(B)}
    $$
    注：由product rule推倒而来.
    </p>

- 多变量的条件概率
    <p>

    $$
    p(A_1, A_2, ..., A_n|B) = p(A_1|B)p(A_2|A_1, B)p(A_3|A_1, A_2, B)...p(A_n|A_1,A_2,...,A_{n-1},B)
    $$
    推导过程（链式计算），
    $$
    p(A_1, A_2, ..., A_n|B) = \frac{p(A_1,A_2,...,A_n,B)}{p(B)} = \frac{p(A_n|A_1,A_2,...,A_{n-1},B)p(A_1,A_2,...,A_{n-1},B)}{p(B)}
    $$
    $$
    =  \frac{p(A_n|A_1,A_2,...,A_{N-1},B)p(A_{n-1}|A_1,A_2,...,A_{n-2},B)p(A_1,A_2,...,A_{n-2},B)}{p(B)}
    $$
    $$
    = \frac{p(A_n|A_1,A_2,...,A_{N-1},B)p(A_{n-1}|A_1,A_2,...,A_{n-2},B)~...~ p(A_1|B)p(B)}{p(B)}
    $$
    $$
    = p(A_1|B)p(A_2|A_1,B)...p(A_n|A_1, A_2,...,A_{n-1}, B)
    $$
    引申出的另外一个公式
    $$
    p(A,B|C) = p(A|B,C)p(B|C)
    $$
    </p>

- 条件独立
    如果在给定C的情况下，A和B是条件独立的，那么，
    <p>

    $$
    p(A, B|C) = p(A|C)p(B|C)
    $$

    $$
    p(A|B,C) = P(A|C)
    $$
    第二个式子的推导,
    $$
    p(A|B,C) = \frac{p(A,B|C)}{p(B|C)} = \frac{p(A|C)p(B|C)}{p(B|C)} = p(A|C)
    $$

    </p>

***Reference:***
- [Introduction to Probability & Statistics: Conditional Independence (*Book)](https://www.probabilitycourse.com/chapter1/1_4_4_conditional_independence.php)
- [Pattern Recognition and Machine Learning: Introduction](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

# SVM
## 基础
**SVM的核心思想**： 找到一个超平面$\textbf{w}^T\textbf{x} + b = 0$将两类数据分离，且数据距离超平面有一定的间隔（margin）。<br>
**推导与理解**：样本$\textbf{x}^{(i)}$到超平面$\textbf{w}^T\textbf{x} + b = 0$的距离为
<p>

$$
\gamma_i = \frac{y_i(\textbf{w}^T\textbf{x}^{(i)} + b)}{||\textbf{w}||}
$$

几何间隔(Geometric Margin)为所有样本点中到超平面距离最小值（SVM目的在于让这个最近的距离最大化）：
$$
\gamma = \min\limits_{i=1,2,...,N}\frac{y_i(\textbf{w}^T\textbf{x}^{(i)} + b)}{||\textbf{w}||}
$$

函数间隔(Functional Margin): 所有样本点中到超平面的未经规范化的距离的最小值
$$
\hat{\gamma_i} = y_i(\textbf{w}^T\textbf{x}^{(i)} + b)
$$

$$
\hat{\gamma} = \min\limits_{i=1,2,...,N}\hat{\gamma_i}
$$

</p>

因此，SVM目标函数为
<p>

$$
\max\limits_{\textbf{w}, b} \gamma
$$
$$
s.t.{~~~} \frac{y_i(\textbf{w}^T\textbf{x}^{(i)} + b)}{||\textbf{w}||} \ge \gamma, ~~i = 1, 2,...,N
$$

因为$\gamma = \frac{\hat{\gamma}}{||\textbf{w}||}$,所以目标函数可转化为<br>
$$
\max\limits_{\textbf{w}, b} \frac{\hat\gamma}{||\textbf{w}||}
$$
$$
s.t.{~~~} y_i(\textbf{w}^T\textbf{x}^{(i)} + b)\ge\hat\gamma, ~i=1,2,...,N
$$

因为函数间隔$\hat\gamma$不影响最优化问题的解，假设将$\textbf{w}$和$b$按比例变为$\lambda\textbf{w}$和$\lambda b$，则函数间隔为$\lambda\hat\gamma$。函数间隔的这一改变并不影响上面最优化问题的不等数约束，也不影响目标函数优化的结果，即变化前后是等价的最优化问题。所以，可以令$\hat\gamma=1$。
<br>
<b>理解</b>: 换个角度思考下，因为成比例$\lambda$改变$\textbf{w}, b$，$\hat\gamma$也会成比例改变，所以如果取$\lambda = \frac{1}{\hat\gamma}$，最优化问题也是不变的。这样就可以理解为什么令$\hat\gamma=1$是可以的了。
<br>
则目标函数可以写为，
$$
\max\limits_{\textbf{w}, b} \frac{1}{||\textbf{w}||}
$$
$$
s.t.{~~~} y_i(\textbf{w}^T\textbf{x}^{(i)} + b)\ge1, ~i=1,2,...,N
$$
转化为最小化问题为，
$$
\min\limits_{\textbf{w}, b}\frac{1}{2}||\textbf{w}||^2
$$
$$
s.t.{~~~} y_i(\textbf{w}^T\textbf{x}^{(i)} + b)\ge1, ~i=1,2,...,N
$$
(注：$||\textbf{w}||^2 = \textbf{w}^T\textbf{w}$)
<br>
该优化问题为凸优化问题(有约束的)。
<br>
<b>凸优化问题</b>: 指约束优化问题
$$
\min\limits_{w} f(w)
$$
$$
s.t.{~~~} g_i(w) \le 0, {~~}i=1,2,...,k
$$
$$
{~~~~~~~~} h_i(w) = 0, {~~}i=1,2,...,l
$$
其中，目标函数$f(w)$和约束函数$g_i(w)$都是$\textbf{R}^n$上连续可微的凸函数，约束函数$h_i(w)$是$\textbf{R}^n$上的仿射函数。
<br>
(注：当函数满足$f(x) = ax + b,~a\in\textbf{R}^n,~b\in\textbf{R}^n,~x\in\textbf{R}^n$时，其为仿射函数。即函数为一次函数，对应空间中对向量线性变换再平移。)
</p>

**最优化求解**: 利用拉格朗日对偶性(Lagrange Duality)。在约束最优化问题中，常常利用对偶性将原始问题转化为对偶问题，通过解对偶问题而得到原始问题的解。
<br>
**原始问题**: 假设$f(x), c_i(x), h_j(x)$是定义在$\textbf{R}^n$上的连续可微函数，考虑约束最优化问题
<p>

$$
\min\limits_{x} f(x)
$$
$$
s.t.{~~~} c_i(x) \le 0, {~~}i=1,2,...,k
$$
$$
{~~~~~~~~~~} h_j(x) = 0, {~~}j=1,2,...,l
$$
此约束最优化问题为原始问题。
<br>
引入拉格朗日函数(Generalized Lagrange Function),
$$
L(x, \alpha, \beta) = f(x) + \sum_{i=1}^{k}\alpha_ic_i(x) + \sum_{j=1}^{l}\beta_jh_j(x)
$$
这里，$\textbf{x} = (x_1, x_2, ..., x_n)^T \in \textbf{R}^n$, $\alpha_i, \beta_j$是拉格朗日乘子，$\alpha_i \ge0$，x的函数
$$
\theta_P = \max\limits_{\alpha, \beta: \alpha_i\ge0}L(x, \alpha, \beta)
$$
下标$P$表示原始问题。
<br>
因为$c_i(x)\le0, h_j(x)=0, \alpha_i\ge0$, 所以在满足约束条件的情况下,
$$
\theta_P(x) = \max\limits_{\alpha, \beta: \alpha_i\ge0}L(x, \alpha, \beta) = f(x)
$$
则优化问题表示为,
$$
\min\limits_{x}\theta_P(x) = \min\limits_{x}\max\limits_{\alpha, \beta: \alpha_i\ge0}L(x, \alpha, \beta)
$$
<b>对偶问题</b>:
$$
\theta_D = \max\limits_{x}L(x, \alpha, \beta)
$$
$$
\max\limits_{\alpha, \beta: \alpha_i\ge0}\theta_D(\alpha, \beta) = \max\limits_{\alpha, \beta: \alpha_i\ge0}\min\limits_{x}L(x, \alpha, \beta)
$$
<br>
<b>定理C.1</b>: 若原始问题和对偶问题都有最优值，则
$$
d^\star = \max\limits_{\alpha, \beta: \alpha_i\ge0}\min\limits_{x}L(x, \alpha, \beta) \le \min\limits_{x}\max\limits_{\alpha, \beta: \alpha_i\ge0}L(x, \alpha, \beta) = p^\star
$$
因为， 对任意的$\alpha, \beta, x$，有
$$
\theta_D(\alpha, \beta) = \min\limits_{x}L(x, \alpha, \beta) \le L(x, \alpha, \beta) \le \max\limits_{\alpha, \beta: \alpha^i\ge0}L(x, \alpha, \beta) = \theta_P(x)
$$
<br>
即，对偶问题的最优值(max value)始终小于等于原始问题的最优值(min value)。
<br>
当满足KKT条件时，
$$
d^\star = p^\star
$$
</p>

**SVM最优解**:
<p>

$$
L(\textbf{w}, b, \alpha) = \frac{1}{2}||\textbf{w}||^2 + \sum_{i=1}^{N}\alpha_i(1 - y_i(\textbf{w}^T\textbf{x}^{(i)} + b)) + \sum_{i}^{N}\alpha_i
$$
原问题，
$$
\min\limits_{\textbf{w}, b}\max\limits_{\alpha}L(\textbf{w}, \alpha, \beta)
$$
对偶问题，
$$
\max\limits_{\alpha}\min\limits_{\textbf{w}, b}L(\textbf{w}, \alpha, \beta)
$$
求解，
(1). 求$\min\limits_{\textbf{w}, b}L(\textbf{w}, b, \alpha)$, 分别令$L(\textbf{w}, b, \alpha)$对$\textbf{w}$和$b$的偏导数为0
$$
\nabla_{\textbf{w}}L(\textbf{w}, b, \alpha) = \textbf{w} - \sum_{i=1}^{N}\alpha_iy_i\textbf{x}^{(i)} = 0
$$
$$
\nabla_{\textbf{b}}L(\textbf{w}, b, \alpha) = \sum_{i=1}^{N}\alpha_iy_i = 0
$$
解得,
$$
\textbf{w} = \sum_{i=1}^{N}\alpha_i y_i \textbf{x}^{(i)}
$$
$$
\sum_{i=1}^{N}\alpha_iy_i = 0
$$
将上述结果带入$L(\textbf{w}, b, \alpha)$中得，
$$
L(\textbf{w}, b, \alpha) = \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j \langle\textbf{x}^{(i)}, \textbf{x}^{(j)}\rangle - \sum_{i=1}^{N}\alpha_i y_i(\langle\sum_{j=1}^{N}\alpha_i y_i \textbf{x}^{(j)}, \textbf{x}^{(i)}\rangle + b) + \sum_{i=1}^{N}\alpha_i
$$
$$
= -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j\langle \textbf{x}^{(i)}, \textbf{x}^{(j)}\rangle + \sum_{i=1}^{N}\alpha_i
$$
即，
$$
\min\limits_{\textbf{w}, b}L(\textbf{w}, b, \alpha) = -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j\langle \textbf{x}^{(i)}, \textbf{x}^{(j)}\rangle + \sum_{i=1}^{N}\alpha_i
$$
(2). 求$\min\limits_{\textbf{w}, b}L(\textbf{w}, b, \alpha)$对$\alpha$的极大值
$$
\max\limits_{\alpha} -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j\langle \textbf{x}^{(i)}, \textbf{x}^{(j)}\rangle + \sum_{i=1}^{N}\alpha_i
$$
$$
s.t.{~~} \sum_{i=1}^{N}\alpha_i y_i = 0
$$
$$
{~~~~~~~~~~~~~~~~~~~~~~~~}\alpha_i \ge 0, ~i=1,2,...,N
$$
转化为最小化问题，
$$
\min\limits_{\alpha} \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j\langle \textbf{x}^{(i)}, \textbf{x}^{(j)}\rangle - \sum_{i=1}^{N}\alpha_i
$$
$$
s.t.{~~} \sum_{i=1}^{N}\alpha_i y_i = 0
$$
$$
{~~~~~~~~~~~~~~~~~~~~~~~~}\alpha_i \ge 0, ~i=1,2,...,N
$$
优化该目标函数，一般采用SMO（Sequential minimal optimization）优化方法，可得$\alpha$的最优解$\alpha^\star = (\alpha_1^\star, \alpha_2^\star,...,\alpha_N^\star)^T$
<br>
根据KKT条件，
$$
\nabla_\textbf{w}L(\textbf{w}^\star, b\star, \alpha^\star) = \textbf{w}^\star - \sum_{i=1}^{N}\alpha_i^\star y_i \textbf{x}^{(i)} = 0
$$
$$
\nabla_bL(\textbf{w}^\star, b\star, \alpha^\star) = -\sum_{i=1}^{N}\alpha^\star y_i = 0
$$
$$
\nabla_\alpha L({\textbf{w}^\star}^T, b\star, \alpha^\star) = \alpha_i^\star(y_i({\textbf{w}^\star}^T\textbf{x}^{(i)} + b) - 1) = 0, ~ i = 1,2,..,N
$$
$$
y_i({\textbf{w}^\star}^T\textbf{x}^{(i)} + b) - 1 \ge 0, ~ i = 1,2,...,N
$$
$$
\alpha^\star \ge 0, ~ i = 1,2,...,N
$$
<b>结果：</b>存在$\alpha^\star = (\alpha_1^\star, \alpha_2^\star,...,\alpha_N^\star)^T$且存在下标为$j$的$\alpha_j^\star > 0$, 使得
$$
\textbf{w}^\star = \sum_{i=1}^{N}\alpha_i^\star y_i \textbf{x}^{(i)}
$$
$$
b^\star = y_j - \sum_{i=1}^{N}\alpha_i y_i\langle \textbf{x}^{(i)}, \textbf{x}^{(j)}\rangle
$$
</p>

***Reference***
- [统计学习方法 李航](https://book.douban.com/subject/10590856/)
- [StackExchange: How does a Support Vector Machine (SVM) work?](https://stats.stackexchange.com/questions/23391/how-does-a-support-vector-machine-svm-work)
- [知乎：支持向量机中的函数距离和几何距离怎么理解？](https://www.zhihu.com/question/20466147)
- [CS229 Lecture notes: Part V Support Vector Machine](http://cs229.stanford.edu/notes/cs229-notes3.pdf)

## 核函数
当数据是非线性可分时（无法用$\textbf{w}^T\textbf{x} + b$超平面分开），通常需要进行一个非线性变换，将非线性问题转化为线性问题。

Kernel是用来计算在高维特征空间中两个向量内积(dot product)。假设我们有从$ R^n \Rightarrow R^m $的映射$\varphi$, 将向量$\textbf{x}, \textbf{y} y$从特征空间$R^n$映射到$R^m$。在$R^m$中，$\textbf{x}$和$\textbf{y}$的内积为$\varphi(\textbf{x})^T\varphi(\textbf{y} )$。核函数$K$定义为$K(\textbf{x}, \textbf{y}) = \varphi(\textbf{x})^T\varphi(\textbf{y})$。

定义核函数的好处（为什么使用核方法？）在于，核函数提供了在不知道是什么空间和是什么映射$\varphi$时计算内积。

例如：定义polynomial kernel
$$K(\textbf{x}, \textbf{y}) = (1 + \textbf{x}^T\textbf{y})^2$$

$$
\textbf{x}, \textbf{y} \in R^2, \textbf{x} = (x_1, x_2), \textbf{y} = (y_1, y_2)
$$

看起来$K$并没有表示出这是什么映射，但是：

$$
K(\textbf{x}, \textbf{y}) = (1 + \textbf{x}^T \textbf{y})^2 = (1 + x_1y_1 + x_2y_2)^2 = 1 + x_1^2y_1^2 + x_2^2y_2^2 + 2x_1y_1 + 2x_2y_2 + 2x_1x_2y_1y_2
$$

相当于将$\textbf{x}$从$(x_1, x_2)$映射为$(1, x_1^2, x_2^2. \sqrt{2}x_1, \sqrt{2}x_2, \sqrt{2}x_1x_2)$，将$\textbf{y}$从$(y_1, y_2)$映射为$(1, y_1^2, y_2^2. \sqrt{2}y_1, \sqrt{2}y_2, \sqrt{2}y_1y_2)$

即

$$\varphi(\textbf{x}) = \varphi(x_1, x_2) = (1, x_1^2, x_2^2. \sqrt{2}x_1, \sqrt{2}x_2, \sqrt{2}x_1x_2)$$

$$\varphi(\textbf{y}) = \varphi(y_1, y_2) = (1, y_1^2, y_2^2. \sqrt{2}y_1, \sqrt{2}y_2, \sqrt{2}y_1y_2)$$

则可知
$$K(\textbf{x}, \textbf{y}) = (1 + \textbf{x}^T \textbf{y})^2 = \varphi(\textbf{x})^T\varphi(\textbf{y})$$

**理解**：对于一些特征，其在低维空间中是线性不可分，但是如果将其映射到高维空间中，就可能会是线性可分的。处理的步骤为：特征 -> 映射到高维空间(使用映射函数$\varphi$) -> 分类算法(定义loss function，多表达为内积的形式)。采用核函数的优点在于，不必确定具体的映射函数$\varphi$是什么，不必显示的计算每个特征向量在映射到高维空间的表达是什么样的，而可以直接用低维空间的数据(坐标值)去计算得出向量在高维空间中内积的结果。

## SVM常用核函数
- Linear kernel
- Polynomial kernel
    $$
    K(\textbf{x}, \textbf{z}) = (\textbf{x} \cdot \textbf{z} + 1)
    $$

- RBF kernel
    A **radial basis function(RBF)** is a real-valued function whose value depends only on the distance from the origin, so that $\phi(\textbf{x}) = \phi(\textbf{||x||})$; or on the distance from some other point $c$, called a *center*, so that $\phi(\textbf{x}, c) = \phi(||\textbf{x - c}||)$.
    <br>
    One common RBF kernel is Gaussian kernel
    $$
    K(\textbf{x}, \textbf{z}) = exp(-\frac{||\textbf{x} - \textbf{z}||^2}{2\sigma^2})
    $$

**Kernel的选择**
- 如果特征维数很高（维数高往往线性可分），可以采用*Least Square*或者*线性核的SVM*
- 如果特征维数较小，而且样本数量一般，不是很大，可以选用*高斯核的SVM*
- 如果特征维数较小，而样本数量很多，需要手工添加一些feature使其变为第一种情况（虽然特征维数较小，但是样本数量太大，SVM需要计算两两样本特征的内积，高斯核计算量过多，所以不适合使用）。

***Reference***
- [How to intuitively explain what a kernel is?](https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is)
- [知乎：SVM的核函数如何选取？](https://www.zhihu.com/question/21883548)

<br><br>

# PCA
## 数据基础知识
**向量的表示:** 向量$\textbf{x}$是空间$R^n$的一个向量，$\mathbf{A} = (\mathbf{a_1}, \mathbf{a_2}, ..., \mathbf{a_n})$是空间$R^n$中的一组基，则$\mathbf{x}$可以表示为：
<p>

$$
\textbf{x} = A{(x_1, x_2,..., x_n)}^T = x_1\mathbf{a_1} + x_2\mathbf{a_2} + ... + x_n\mathbf{a_n}
$$
则向量$\mathbf{x}$在空间$R^n$中以$\mathbf{A}$为基的坐标为$(x_1, x_2,..., x_n)$.
</p>

**基变换与坐标变换:** $\mathbf{A} = (\mathbf{a_1}, \mathbf{a_2}, ..., \mathbf{a_n})$和$\mathbf{B} = (\mathbf{b_1}, \mathbf{b_2}, ..., \mathbf{b_n})$分别是空间$R^n$中的两组基。向量$\mathbf{x}$是空间$R^n$的一个向量，其在$\mathbf{A}$中的坐标为$(x_1, x_2,..., x_n)$，则其在以$\mathbf{B}$为基时的坐标为：

<p>

根据基变换的相关知识，我们将基$\mathbf{A}$与基$\mathbf{B}$的变换关系描述为，
$$
\mathbf{B} = (\mathbf{b_1}, \mathbf{b_2}, ...,\mathbf{b_n}) = (\mathbf{a_1}, \mathbf{a_2}, ..., \mathbf{a_n})\left[\begin{matrix}
  m_{11} & m_{12} & ... & m_{1n}\\
  m_{21} & m_{22} & ... & m_{1n}\\
  ...    & ...    & ... & ...   \\
  m_{n1} & m_{n2} & ... & m_{nn}
\end{matrix}\right] = \mathbf{A} \mathbf{T}
$$
则向量$\mathbf{x}$在基$\mathbf{B}$中的坐标为：
$$
\mathbf{x} = \mathbf{A}[\mathbf{x}]_{A} = \mathbf{B} [\mathbf{x}]_{B}
$$
$$
\mathbf{A}[\mathbf{x}]_{A} = \mathbf{AT} [\mathbf{x}]_{B}
$$
$$
[\mathbf{x}]_{B} = \mathbf{T}^{-1}[\mathbf{x}]_{A}
$$
其中，$\lbrack\mathbf{x}\rbrack_{A}$表示向量$\mathbf{x}$在基$\mathbf{A}$中的坐标。
</p>

***Reference:***
- [豆瓣：矩阵基的变换和坐标理解](https://www.douban.com/note/549051760/)
- [可汗学院-线性代数：基变换的矩阵](http://open.163.com/movie/2011/6/7/F/M82ICR1D9_M83J6M77F.html)
- [Tex: Which “bold” style is recommended for matrix notation?](https://tex.stackexchange.com/questions/199789/which-bold-style-is-recommended-for-matrix-notation)

## PCA原理
**思想：** 在信号处理中，人们认为噪声的方差较小，而有用信息的方差较大。因此我们希望找到一种变换，将原始空间中的基变换为另一个空间的一组可以有效描述有用信息的基，即在新基下，样本在基的方向上方差最大。<br>
**理解：** 在原始空间中，找到样本方差最大的方向（样本都投影到该方向上，方差最大，且样本距离这一方向的直线/超平面的距离最小），用该方向作为新空间中的基的一个。<br>

在空间$R^n$中，$m$组样本构成的矩阵$\mathbf{X} = \left[\begin{matrix}
  \mathbf{x_1}\\
  \mathbf{x_2}\\
  ...\\
  \mathbf{x_n}
\end{matrix}\right]$, 其中${\mathbf{x^{(i)}}} = (x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)})$。<br>
首先，需对$\mathbf{X}$进行去均值处理，即对$m$维的列向量分别计算均值并减去均值，并进行标准化。<br>
之后，为了找到方差最大的方向$\mathbf{u}$作为空间的基（最好为单位向量），需要计算
<p>

$$
\max\limits_{\mathbf{\mu}} \frac{1}{m}\sum_{i = 1}^m {(\langle \mathbf{x}^{(i)}, \mathbf{u} \rangle - \mu)}^2
$$
$$
s.t. ~~~ {\mathbf{u}}^T \mathbf{u} = 1
$$
因为样本的期望$\mu = 0$, 并且，
$$
\langle \mathbf{x^{(i)}}, \mathbf{u} \rangle = {\mathbf{x}^{(i)}}^T \mathbf{u} = {\mathbf{u}}^T {\mathbf{x}^{(i)}}
$$
所以，
$$
\frac{1}{m}\sum_{i = 1}^m {(\langle \mathbf{x}^{(i)}, \mathbf{u} \rangle - \mu)}^2 =  \frac{1}{m}\sum_{i=1}^{m}{\mathbf{u}}^T {\mathbf{x}^{(i)}} {\mathbf{x}^{(i)}}^T \mathbf{u} = \mathbf{u}^T (\frac{1}{m}\sum_{i=1}^{m}\mathbf{x}^{(i)} {\mathbf{x}^{(i)}}^T) \mathbf{u}
$$
即，
$$
\max\limits_{\mathbf{\mu}} \mathbf{u}^T (\frac{1}{m}\sum_{i=1}^{m}\mathbf{x}^{(i)} {\mathbf{x}^{(i)}}^T) \mathbf{u}
$$
$$
s.t. ~~~ {\mathbf{u}}^T \mathbf{u} = 1
$$
采用拉格朗日乘数法，
$$
L(\mathbf{u}, \lambda) = \mathbf{u}^T (\frac{1}{m}\sum_{i=1}^{m}\mathbf{x}^{(i)} {\mathbf{x}^{(i)}}^T) \mathbf{u} - \lambda({\mathbf{u}}^T\mathbf{u} - 1)
$$
记$\mathbf{\Sigma} = \frac{1}{m}\sum_{i=1}^{m}\mathbf{x}^{(i)} {\mathbf{x}^{(i)}}^T$，且${\mathbf{\Sigma}}^T = \mathbf{\Sigma}$
$$
\nabla_{\mathbf{u}} L = (\mathbf{\Sigma} + {\mathbf{\Sigma}}^T) \mathbf{u} - 2\lambda \mathbf{u} = 2\mathbf{\Sigma}\mathbf{u} - 2\lambda \mathbf{u} = 0  
$$
即，
$$
\mathbf{\Sigma}\mathbf{u} = \lambda \mathbf{u}
$$
因此，$\lambda$为矩阵$\mathbf{\Sigma}$的特征值，$\mathbf{\Sigma}$为向量变量$\mathbf{x}$的协方差矩阵，也可以用样本矩阵$\mathbf{X}$计算出$\mathbf{\Sigma} = \frac{1}{m}\mathbf{X}^T\mathbf{X}$。
<br>
<b>注</b>： $\frac{\partial \mathbf{x}^T \mathbf{A} \mathbf{x}}{\mathbf{x}} = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$<br>

原始的样本向量${\mathbf{x}}^{(i)} \in R^n$，变换完后（降维后的）的向量记做${\mathbf{y}}^{(i)}$,
$$
\mathbf{y}^{(i)} = {(\mathbf{u_1}^T\mathbf{x}^{(i)}, \mathbf{u_2}^T\mathbf{x}^{(i)}, ..., \mathbf{u_k}^T\mathbf{x}^{(i)})}^T
$$
<b>个人的理解：</b> 从线性变换的角度上理解PCA的原理就是说，找到一种变换，将原空间的基变换为另一组，其样本在基的方向上方差最大，之后选取前$k$个基组成新的低维的空间。因为样本在这一组基上的方差大，所以尽可能多的保留了信息（噪声的方差小，被省去了），从而达到了降维的目的。其中对应的线性变换的矩阵为$\mathbf{\Sigma} = (\mathbf{u_1}, \mathbf{u_2}, ..., \mathbf{u_n})$，即$\mathbf{B} = \mathbf{A}\mathbf{\Sigma}$。则在两个不同基下的向量坐标为
$$
\lbrack \mathbf{x}^{(i)}\rbrack_B = \mathbf{\Sigma}^{-1} \lbrack \mathbf{x}^{(i)}\rbrack_A = \mathbf{\Sigma}^T \lbrack\mathbf{x}^{(i)}\rbrack_A = \left[\begin{matrix}
  \mathbf{u_1}^T\\
  \mathbf{u_2}^T\\
  ...\\
  \mathbf{u_n}^T
\end{matrix}\right] \lbrack\mathbf{x}^{(i)}\rbrack_A
$$
随后截取前$k$维（投影），便获取了降维后的结果。<br>
这些特征向量$\mathbf{u_i}$是子空间的基，可以线性组合出子空间的所有点，因而可以将这些特征向量作为模版（例如特征脸），去组合出别的样本脸。但要注意的是不要单独考虑其中某一个特征向量，应整体考虑，因为他们是子空间的一组基，单独考虑特别的意义。—— by Andrew Ng
</p>

**PCA的应用**
- 可视化：将数据降维到2，3维，方便绘制出可视化图像
- 数据压缩
- 机器学习：高维数据通常会落在低维子空间中（高维空间中的很多点是无意义的），即降维不会丢失数据间的pattern
- 异常检测
- 距离计算：不在高维空间中计算两个样本的距离，转换到低维空间进行计算

***Reference:***
- [机器学习中的数学(4)-线性判别分析（LDA）, 主成分分析(PCA)](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html)
- [网易云课堂：cs229——主成分分析法](http://open.163.com/movie/2008/1/M/E/M6SGF6VB4_M6SGKIEME.html)
- [机器学习中常用的矩阵求导公式](http://www.voidcn.com/article/p-ponrkmdt-xd.html)
- [The Matrix Cookbook - Mathematics](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
- [KaTex: how to output a matrix? #667](https://github.com/Khan/KaTeX/issues/667)
<br>

## PCA与SVD
如果当特征维数很高时，其协方差矩阵的维度也会很高。例如，$\mathbf{x} \in R^{5000}$，则且协方差矩阵$\mathbf{\Sigma} \in R^{5000 * 5000}$。因此，直接计算其特征值和特征向量是很花费时间的。<br>

**奇异值分解(Singular Value Decompotition, SVD):** 矩阵$\mathbf{A} \in R^{m * n}$, 则其可以被分解为，
<p>

$$
\mathbf{A} = \mathbf{U} \mathbf{D} \mathbf{V}^T
$$
其中，$\mathbf{U} \in R^{m * n}, \mathbf{D} \in \mathbf{n * n}, \mathbf{V} \in R^{n * n}$，$\mathbf{D}$为对角矩阵(Diagonal Matrix),
$$
\mathbf{D} = \left[\begin{matrix}
  \sigma_1\\
  &\sigma_2\\
  &&...\\
  &&&\sigma_n
\end{matrix}\right]
$$
$\sigma_i$称矩阵$\mathbf{A}$的奇异值。<br>
$\mathbf{U}$的列向量是矩阵$\mathbf{A}\mathbf{A}^T$的特征向量，$\mathbf{V}$的列向量是矩阵$\mathbf{A}^T\mathbf{A}$的特征向量。<br>

</p>
<br>
<p>
<b>SVD与PCA：</b>

因为，向量$\mathbf{x}$（每一维对应了一个特征变量）的协方差矩阵$\mathbf{\Sigma}$可以根据其样本样本矩阵计算出，即$\mathbf{\Sigma} = \mathbf{X}^T\mathbf{X}$，则，
$$
\mathbf{X} = \mathbf{U} \mathbf{D} \mathbf{V}^T
$$
$$
\mathbf{\Sigma} = \mathbf{X}^T \mathbf{X} = \mathbf{V} \mathbf{D} \mathbf{U}^T \mathbf{U} \mathbf{D} \mathbf{V}^T = \mathbf{V} \mathbf{D}^2 \mathbf{V}^T
$$
因为$\mathbf{V}$是正交矩阵，所以$\mathbf{V}^{-1} = \mathbf{V}^T$，所以上式为矩阵$\mathbf{\Sigma}$的特征值分解，则$\mathbf{V}$的列向量（也是矩阵$\mathbf{U}$的行向量）是$\mathbf{\Sigma}$的特征向量，而$\mathbf{X}$的奇异值的平方是$\mathbf{\Sigma}$的特征值，此时通过对$\mathbf{X}$的奇异值分解，而得到了其PCA的计算，同时避免了直接计算高维矩阵$\mathbf{\Sigma}$的特征值与特征向量。
</p>

***Reference:***
- [网易公开课：cs229——奇异值分解](http://open.163.com/movie/2008/1/J/V/M6SGF6VB4_M6SGKINJV.html)