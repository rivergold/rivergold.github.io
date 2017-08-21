# SVM
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
