# Basics

## Standard notations
- Variable: $X$ (uppercase and no bold)
- Matrix: $\mathbf{X}$ (upper-case and bold)
- Vetor: $\mathbf{x}$ (lower-case and bold)
- Element/Scalar: $x$ (lower-case and no bold)

<br>
<br>

## Basic Steps for Deep Learning
1. Define the model structure
2. Initialize the model's parameters
3. Loop:
    - Calculate current loss(forward propagation)
    - Calculate current gradient(backward propagation)
    - Update parameters(gradient descent)

<br>
<br>

## Backpropagation

<p align="center">
    <img src="http://ovvybawkj.bkt.clouddn.com/dl/backpropagation_1.png" width="75%">
</p>

<p>

Here are some notations we will need later. We use $w^l_{jk}$ to denote the weight for the connection from the $k^{th}$ neuron in the $(l - 1)^{th}$ layer to the $j^{th}$ neuron in the $l^{th}$ layer. And we use $z^{l}_j$ to represent the input of the $j^{th}$ neuron in the $l^{th}$ layer, $a^l_j$ to represent the activation output in the j^{th} neuron in the $l^{th}$ layer. Similarly, $b^l_j$ represents the bias of the $j^{th}$ neuron in the $l^{th}$ layer.<br>
<br>
<b>Why use this cumbersome notation?</b> Maybe it is better to use $j$ to refer to the inpurt neuron, and $k$ to the output neuron. Why we use vice versa? The reason is the activation output of the $j^{th}$ neuron in the $l^{th}$ layer can be expressed like,

$$
a^l_j = \sigma(\sum_k w^l_{jk}a^{l-1}_k + b^l_j)
$$

This expression can be rewritten into a matrix from as followings,
$$
\mathbf{a}^l = \sigma(\mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l)
$$
where, $\mathbf{a}^{l}$, $\mathbf{a}^{l-1}$ and $\mathbf{b}^l$ are vectores, $\mathbf{W}^l$ is a <b>weight matirx</b> for the $j^{th}$ layer, and its $j^{th}$ row and $k^{th}$ column is $w^l_{jk}$. The elements in $j^{th}$ row of $\mathbf{W}^l$ are reprent the weights of neurons in $(l-1)^{th}$ layer connecting to the $j^{th}$ neuron in $l^{th}$ layer.<br><br>

Then, we define the loss function $C$, here we use the following notation(mean square error, MSE) as a example,
$$
C = \frac{1}{2}\frac{1}{m}\sum_{i}^{m}\| \mathbf{y}^{(i)} - \mathbf{a}^{L}(\mathbf{x}^{(i)}) \|^2
$$
where, $L$ denotes the number of layers in the networks, $\mathbf{a}^L$ denotes the final output of the network. And the loss of a single training example is $C_{\mathbf{x}^{(i)}} = \frac{1}{2}\|\mathbf{y}^{(i) - \mathbf{a}^L}\|^2$.<br><br>

<b>Note:</b> Backpropagation actually compute the partial derivatives $\frac{\partial C_{x^{(i)}}}{\partial w}$ and $\frac{\partial C_{x^{(i)}}}{\partial b}$ for single trainning example. Then, we calculate $\frac{\partial C}{\partial w}$ and $\frac{\partial C}{\partial b}$ by averageing over training samples (this step is for GD or mini-bath GD). Here we suppose the training example $\mathbf{x}$ has been fixed. And in order to simplify notation, we drop the $\mathbf{x}$ subscript, writing the loss $C_\mathbf{x}^{(i)}$ as $C$.<br>

So, for each single training sample $\mathbf{x}$, the lose maybe written as,
$$
C = \frac{1}{2}\| \mathbf{y} - \mathbf{a}^L \| = \frac{1}{2}\sum_j (y_j - a^L_j)^2
$$
Here, we define $\delta^l_j$ as
$$
\delta^l_j = \frac{\partial C}{\partial z^l_j}
$$

</p>

$\delta^l_j$ shows that the input of $j^{th}$ neuron in the $l^{th}$ layer influences the extent of the network loss change (Details can be obtained from [here](http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation)).<br>

<p>

<b>理解：</b>$\delta^l_j$表达了在第$l$层网络的第$j$个神经元的输入值的变化对最终的loss function的影响程度。<br>

And we have,
$$
z^l_j = \sum_k w_{jk}^l a_k^{l-1}
$$
$$
a_j^l = \sigma(z_j^l)
$$
Then,
$$
\delta_j^L = \frac{\partial C}{\partial z^L_j} = \sum_k \frac{\partial C}{\partial a^L_k} \frac{\partial a_k^L}{\partial z_j^L} = \frac{\partial C}{\partial a^L_k} \sigma^{'}(z^L_j)
$$
Moreover,
$$
\delta^l_j = \frac{\partial C}{\partial z_j^l} = \sum_k \frac{\partial C}{\partial z^{l + 1}_k} \frac{\partial z_k^{l+1}}{\partial z_j^l} = \sum_k \delta^{l+1}_k \frac{\partial z_k^{l+1}}{\partial z_j^l}
$$
Because
$$
z_k^{l+1} = \sum_i w_{ki}^{l+1}a_i^l + b^l_i = \sum_i w^{l+1}_{ki}\sigma(z^{l}_i) + b^l_i
$$
Differentiating, we obtain
$$
\frac{\partial z_k^{l+1}}{\partial z_j^l} = w^{l+1}_{kj}\sigma^{'}(z^l_j) ~~~~~~(i = j)
$$
Then, we get
$$
\delta_j^l = \sum_k \delta_k^{l+1} w^{l+1}_{kj} \sigma^{'}(z^l_j)
$$
<br>
<b>理解：</b>$w^{l+1}_{kj}$表示位于$(l+1)^{th}$层的$k^{th}$神经元连接到$l^{th}$层$j^{th}$神经元的权值，该公式表明，将$(l+1)^{th}$层的所有神经元的梯度变化分别乘以其与$l^{th}$层$k^{th}$神经元的权值并相加。<br>
Our goal is to update $w^l_{jk}$ and $b^l_j$, and we need to calculate the partial derivative,
$$
\frac{\partial C}{\partial w_{jk}^{l}} = \sum_i \frac{\partial C}{\partial z^l_{i}} \frac{\partial z^l_i}{w^l_{jk}} = \frac{\partial C}{\partial z^l_{j}} \frac{\partial z^l_{j}}{\partial w^l_{jk}} = \delta^{l}_j a^{l-1}_k
$$
$$
\frac{\partial C}{\partial b^l_j} = \sum_i \frac{\partial C}{z^l_i} \frac{\partial z^l_i}{b^l_j} = \delta_j
$$
So far, we have four key formulas of backpropagation,
$$
\begin{aligned}
& \delta_j^L = \frac{\partial C}{\partial a^L_k} \sigma^{'}(z^L_j) & ~(1) \\
& \delta_j^l = \sum_k \delta_k^{l+1} w^{l+1}_{kj} \sigma^{'}(z^l_j) & ~(2) \\
& \frac{\partial C}{\partial w_{jk}^{l}} = \delta^{l}_j a^{l-1}_k &~(3)\\
& \frac{\partial C}{\partial b^l_j} = \delta_j^l &~(4) \\
\end{aligned}
$$

</p>

<br>

### Deduce BP with Vectorization
Here we use the concept of differential:
- Monadic calculus: $\mathrm{d}f = f^{'}(x)\mathrm{d}x$
- **Multivariable calculus**:
    - Scalar to vector
    <p>

    $$
    \mathrm{d}f = \sum_i \frac{\partial f}{\partial x_i} = {\frac{\partial f}{\partial \mathbf{x}}^T}\mathrm{d}\mathbf{x}
    $$

    </p>

    - Scalar to matrix
    <p>

    $$
    \mathrm{d}f = \sum_{i, j}\frac{\partial f}{x_{ij}}\mathrm{d}x_{ij} = \mathrm{tr}({\frac{\partial f}{\partial \mathbf{X}}}^T)\mathrm{d}\mathbf{X}
    $$

    </p>

So, we can get,
<p>

$$
\frac{\partial J}{\partial {\mathbf{W}}^L} = \frac{\partial J}{\partial \mathbf{a}^L} \frac{\partial \mathbf{a}^L}{\partial \mathbf{z}^L} \frac{\partial \mathbf{z}^L}{\partial \mathbf{W}^L}
$$

$$
\frac{\partial J}{\partial \mathbf{W}^{L-1}} = \frac{\partial J}{\partial\mathbf{a}^L} \frac{\partial \mathbf{a}^L}{\partial \mathbf{z}^L} \frac{\partial \mathbf{z}^L}{\mathbf{a}^{L-1}} \frac{\partial\mathbf{a}^{L-1}}{\partial \mathbf{z}^{L-1}} \frac{\partial\mathbf{z}^{L-1}}{\partial\mathbf{W}^{L-1}}
$$

$$
...
$$

$$
\frac{\partial J}{\partial \mathbf{W}^l} = \frac{\partial J}{\mathbf{z}^l} \frac{\partial \mathbf{z}^l}{\partial \mathbf{W}^l}
$$

$$
\frac{\partial J}{\partial \mathbf{b}^l} = \frac{\partial J}{\partial \mathbf{z}^l}
$$

Note that,
$$
\mathrm{d}\mathbf{z}^l = \mathrm{tr}(\mathrm{d}\mathbf{W}^l\mathbf{a}^{l-1}) = \mathrm{tr}(\mathbf{a}^{l-1}\mathrm{d}\mathbf{W}^{l})
$$

$$
\frac{\partial \mathbf{z}^l}{\partial \mathbf{W}^l} = {\mathbf{a}^{l-1}}^T
$$

$$
\frac{\partial J}{\partial\mathbf{W}^l} = \frac{\partial J}{\partial\mathbf{z}^l} {\mathbf{a}^{l-1}}^T
$$

And we can rewrite these formulas into matrix-based form, as
$$
\begin{aligned}
& \delta^L = \nabla_{\mathbf{a}^L} C \odot \sigma^{'}(\mathbf{z}^L) & ~(1) \\
& \delta^l = ({(\mathbf{W}^{l+1})}^T \delta^{l+1}) \odot \sigma^{'}(\mathbf{z}^l) & ~(2) \\
& \nabla_{\mathbf{W}^l}C = \delta^{l} {(\mathbf{a}^{l - 1})}^T & ~(3) \\
& \nabla_{\mathbf{b}^l}C = \delta^l & ~(4) \\
\end{aligned}
$$

</p>

***Reference:***
- [知乎：矩阵求导术（上）](https://zhuanlan.zhihu.com/p/24709748)
- [Neural Networks and Deep Learning: How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

<br>

## Regularization
### $L_1$ regularization
<p>

$$
\lambda \sum_{i=1}^{n} \| \mathbf{w} \| = \lambda {\|\mathbf{w}\|}_1
$$

</p>

By using $L_1$ regularization, $\mathbf{w}$ will be sparse.

### $L_2$ regularization
$L2$ regularization are used much more often during training neural network, it will make weights uniform,

<p>

$$
\lambda \sum_{i=1}^{n}\|\mathbf{w}\|^2 = \lambda {\|\mathbf{w}\|}_2
$$

</p>

In neural network, the loss function with regularization is written as,

<p>

$$
J(\mathbf{w}^1, b^1, \mathbf{w}^2, b^2, ...,  \mathbf{w}^L, b^L) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum_{l=1}^{L}\|\mathbf{w}^l\|^2_2
$$

where, $\mathbf{w}^l$ is weights matrix and $b$ is a bias vector.<br>
When we do backpropagation to update weights(here assume we use SGD), the gradient of $\mathbf{w}^L$ is,

$$
\frac{\partial J}{\partial \mathbf{w}^L} = \frac{\partial L(\hat{y}^{(i)}, y^{(i)})}{\partial \mathbf{w}^L} + \lambda \mathbf{w}^L
$$
we note $\frac{\partial L(\hat{y}^{(i)}, y^{(i)})}{\partial \mathbf{w}^L}$ as $\mathrm{d}\mathbf{w}^L$
$$
\mathbf{w}^L := \mathbf{w}^L - \alpha \mathrm{d} \mathbf{w}^L - \alpha \lambda \mathbf{w}^L
$$
Here, the $\lambda$ is called **weight decay**, no matter what value of $mathbf{w}^L$ is, this notation is intent to decay the weights(make weights' absolute value small).

</p>

### Dropout
Core concept:
1. Dropout randomly knocks out units in the network, so it's as if on every iteration, we are working with a smaller neural network and so using a smaller neural network seems like it should has a regularization effect.

2. Make the neuron can not rely on any one feature, so it makes to spread out weights.<br>
**理解：** dropout在每一次迭代都会抛弃部分输入数据（使某些输入为0），不使权值集中于某个或者部分输入特征上，而是使权值参数更加均匀的分布，可以理解为**shrink weights**，因此于$L_2$正则化类似。
<br>

Tips of using Dropout:
1. Dropout is for preventing over-fitting. It the model is not over-fitting, it's better not to use dropout.<br>
**理解：** Dropout是用来解决over-fitting的，如果模型没有over-fitting，不必非要使用。

2. Because of dropout, the loss function $J$ can not be defined explicitly. So it's hard to check whether loss decrease rightly. It's a good choice to close dropout and check the loss decreases right to ensure that your code has no bug, and then open dropout to work.

### Other Regularization Methods
1. Data augmentation
    - Flipping
    - Random rotation
    - Clip
    - Distortion

2. Early stopping
Check dev set error and early stop training. $\mathbf{w}$ is small at initialization and it will increase along with iteration. Early stop will get a mid-size rate $\mathbf{w}$, so it's similar to $L_2$ regularization.
<br>

## Preprocessing
Most dateset maybe have different size of image, a common preprocession is:
1. Scale image into same size or scale one side(width or height, often the short one) into the same size
2. Do data augmentation: flipping, random rotation
3. Crop a square from each image randomly
4. mean subtract
### Per-pixel mean subtract
Subtract input image with per-pixel mean. The whole training set is $(N, C, H, W)$, the per-pixel mean is calculated by for each $C$ computing the average of all the same position pixel over all image, and then will get `mean matrix` which size is $(C, H, W)$.
```python
# X size is (N, C, H, W)
mean = np.mean(X, axis=0)
mean.shape
>>> (C, H, W)
``` 
`Caffe` use per-pixel mean subtract in its [tutorial]().<br>
**注：** per-pixel mean处理时，每个通道是独立处理的， 因为不同通道的像素不具有平稳性（图像中不同部分的统计特性是相同的），并对同一位置的像素计算所有样本的平均值。

### Per-channel mean subtract
Subtract the mean of per channel calculated over all images. The training set size is $(N, C, H, W)$, the mean is calculated each channel over all images, and get the `mean vector` size of $(C, )$.
```python
# X size is (N, C, H, W)
mean = np.mean(X, axis=(0, 2, 3))
mean.shape
>>> (C,)
```
<br>

Whether **per-pixel mean subtract** or **per-channel mean subtract**, they all serves to "center" the data, it means to make the mean of the dataset is around zero, which will help train the networks(make gradient healthy). And as far as I knowm, **per-channel mean subtract** is better and common choice for preprocessing.

***References:***
- [Github: KaimingHe/deep-residual-networks: preprocessing? #5](https://github.com/KaimingHe/deep-residual-networks/issues/5)
- [caffe: Brewing ImageNet](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html)
- [Google Groups: Subtract mean image/pixel](https://groups.google.com/forum/#!topic/digits-users/FfeFp0MHQfQ)
- [StackExchange: Why do we normalize images by subtracting the dataset's image mean and not the current image mean in deep learning?](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)
- [MathWorks: What is per-pixel mean?](https://cn.mathworks.com/matlabcentral/answers/292415-what-is-per-pixel-mean)

<br>

## Batch Normalization

<p>

Assume $\mathbf{X}$ is 4d input $(N, C, H, W)$, the output of batch normalization layer is
$$
y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
where $x$ is a mini-batch of 3d input $(N, H, W)$. The $\mathrm{E}[x]$ and $\mathrm{Var}[x]$ are calculate pre-dimension over the mini-batches and $\gamma$ and $\beta$ are learnable parameter vectors of size $C$(the input size).<br><br>

<b>理解：</b>按照$C$的维度，把其他维度的值拉成一个向量计算均值和方差，之后进行归一化：即对每个Channel的所有mini-batch样本所有值计算均值和方差并归一化。

</p>

A toy example:
```python
from torch import nn
from torch.autograd import Variable
import numpy as np

x = np.array([
             [[[1,1], [1,1]], [[1,1], [1,1]]],
             [[[2,2],[2,2]], [[2,2], [2,2]]]
             ], dtype=np.float32)
x = Variable(torch.from_numpy(x))
# No affine parameters.
bn = nn.BatchNorm2d(2, affine=False)
output = bn(x)
>>> Variable containing:
(0 ,0 ,.,.) =
 -1.0000 -1.0000
 -1.0000 -1.0000

(0 ,1 ,.,.) =
 -1.0000 -1.0000
 -1.0000 -1.0000

(1 ,0 ,.,.) =
  1.0000  1.0000
  1.0000  1.0000

(1 ,1 ,.,.) =
  1.0000  1.0000
  1.0000  1.0000
[torch.FloatTensor of size 2x2x2x2]
```

***Reference:***
- [PyTorch: BathNorm2d](http://pytorch.org/docs/master/nn.html#batchnorm2d)
- [pytorch: 利用batch normalization对Variable进行normalize/instance normalize](http://blog.csdn.net/u014722627/article/details/68947016)

<br>

## Weight Initialization
<p>

Input featres $\mathbf{x} \sim \mathcal{N}(\mu, \sigma^2)$，the output $\mathbf{a} = \sum_{i=1}^{n}w_ix_i$，其方差为
$$
\mathrm{Var}(a) = \mathrm{Var}(\sum_{i=1}^{n}w_ix_i) = \sum_{i=1}^{n}\mathrm{Var}(w_ix_i)
$$
$$
= \sum_{i=1}^{n}[\mathrm{E}(w_i)]^2\mathrm{Var}(x_i) + [\mathrm{E}(x_i)]^2 \mathrm{Var}(w_i) + \mathrm{Var}(w_i)\mathrm{Var}(x_i)
$$
$$
= \sum_{i=1}^{n} \mathrm{Var}(w_i) \mathrm{Var}(x_i)
$$
$$
= n\mathrm{Var}(w) \mathrm{Var}(x)
$$

Here, we assumed zero mean inputs and weights, so $\mathrm{E}[x_i] = 0, \mathrm{E}[w_i] = 0$, and $w_i, x_i$ are independent each other, $x_i (i = 1,2,..,n)$ are independent identically distributed and $w_i (i = 1,2,..,n)$ are alse independent identically distributed.<br>
If we want output $a$ to have the same variance as all of its input $x$, the variance of $w$ needs to be $\frac{1}{n}$, $\mathrm{Var}(x) = \frac{1}{n}$, it means $w \sim \mathcal{N}(0, \frac{1}{n})$.<br><br>

<b>理解：</b>我们假设了输入特征和权重的均值都是0，$\mathrm{E}[x_i] = 0$，$\mathrm{E}(w_i) = 0$，并且$w_i, x_i$之间都是相互独立的，且$x_i$独立同分布，$w_i$独立同分布。因此，如果想要$a$与$x$的方差相同（网络输入与输出的分布不发生改变），我们需要让$\mathrm{Var}(w) = \frac{1}{n}$，即$w \sim \mathcal{N}(0, \frac{1}{n})$，又因为$\mathrm{Var}(nx) = n^2\mathrm{Var}(x)$，所以有`w = np.random.randn(n) / sqrt(n)`.
另外，在深度学习代码实现中，通常采用下面所示的方法对参数初始化
</p>

```python
# Calculate standard deviation.
stdv = 1 / math.sqrt(n)
# Numpy
w = np.random.uniform(-stdv, stdv)
```
即在以均值0为中心，一个标准差的范围内进行随机采样，这样使权值$w$更为接近0。

***Reference:***
- [cs231n: Weight Initialization](http://cs231n.github.io/neural-networks-2/#init)
- [Wiki: Variance](https://en.wikipedia.org/wiki/Variance)
- [知乎: 为什么神经网络在考虑梯度下降的时候，网络参数的初始值不能设定为全0，而是要采用随机初始化思想？](https://www.zhihu.com/question/36068411)

<br>

## Optimization Methods
<p>
Loss function is defined as,

$$
J(\mathbf{w}, \mathbf{x}) = \frac{1}{2}\frac{1}{m}\sum_{i=1}^{m}(h_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)})^2 + \frac{1}{2}\lambda \mathbf{w}^2
$$
$$
J(\mathbf{w}, \mathbf{x})^{(i)} = \frac{1}{2}(h_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)})^2 + \frac{1}{2}\lambda\mathbf{w}^2
$$
$$
\nabla_{\mathbf{w}} J(\mathbf{w}, \mathbf{x})^{(i)}
$$

</p >

### Batch Gradient Descent(BGD)
BGD calculate the sum gradients of all samples and get the mean,
<p>

$$
w_i := w_i - \eta \frac{1}{m}\sum_{k=1}^{m}\nabla_{w_i}J(\mathbf{w}, \mathbf{x})^{(k)}
$$

</p>

**Advantages**:
- Simpleness

**Disdvantages**:
- Large amounts of computation
- Memory may not enough to put all samples
- Difficult to update weights online

### Stochastic Gradient Descent
SGD get one sample and calculate the gradient to update weights,
<p>

$$
w_i := w_i - \eta \nabla J(\mathbf{w}, \mathbf{x})^{(k)}
$$

</p>

### Mini-batch Gradient Descent
These method calculate the gradient of a mini-batch samples and the get the mean to update weights,
<p>

$$
w_i := w_i - \eta \frac{1}{b}\sum_{k=j}^{j+b}\nabla_{w_i}J(\mathbf{w}, \mathbf{x})^{(k)}
$$

</p>

The following methods are optimized based on **Gradient Descent**, we ues $g$ to notate gradient $\nabla J(\mathbf{w}, \mathbf{x})$(This gradient can be the mean of all samples, a samples or a mean of a batch of samples). And in deep learning we often use SGD, but you should know that SGD here represents mini-batch gradient descent.

### Momentum

<p>

$$
v := \mu v_{t-1} + g
$$
$$
w_i := w_i - \eta v
$$

</p>

### Nesterov Momentum
<p>

$$
v := \mu v_{t-1} + g
$$
$$
w_{i_{\text{next}}} = w - \eta v
$$
$$
v := \mu v_{t-1} + g_{w_{i_{\text{next}}}}
$$
$$
w_i := w_i - \eta v
$$

</p>

***Reference:***
- [知乎专栏：深度学习最全优化方法总结比较（SGD，Adagrad，Adadelta，Adam，Adamax，Nadam）](https://zhuanlan.zhihu.com/p/22252270)
- [卷积神经网络中的优化算法比较](http://shuokay.com/2016/06/11/optimization/) (注：该博客写的有些错误，主要了解其讲解的思想)
- [知乎：在神经网络中weight decay起到的做用是什么？momentum呢？normalization呢？](https://www.zhihu.com/question/24529483)

## 1D, 2D, 3D Convlutions
- 1D convolution:
    - Input: a vector $[C_{in}, L_{in}]$
    - Kernel: a vector $[k,]$
    - Output(one kernel): a vector $[L_{out},]$
- 2D convolution:
    - Input: a image $[1, H, W]$ or $[C_{in}, H, W]$
    - Kernel: $[C_{in}, k, k]$
    - Output(one kernel): a feature map $[H_{out}, W_{out}]$
- 3D convolution:
    - Input: a video or CT $[C_{in}, D, H, W]$
    - Kernel: $[C_{in}, k, k, k]$
    - Output(one kernel): $[D_{out}, H_{out}, W_{out}]$

Notice that the dimensions of the output after convolution make the name of what kind convolution it is.<br>
**注：** 几维的卷积是由一个卷积核卷积之后的输出结果的维度决定的。

***References:***
- [网易-deeplearning.ai: Convolution over volumes](https://mooc.study.163.com/learn/deeplearning_ai-2001281004?tid=2001392030#/learn/content?type=detail&id=2001728687&cid=2001725124)

## Loss Function
### Classification
#### Cross Entropy
<p>

$$
H(Y, \hat{Y}) = E_Y [\frac{1}{\log \hat{Y}}] = E_Y [-\log \hat{Y}]
$$

</p>

**Basic knowledge:**
- **Entropy(Shannon Entropy):** Shannon defined the entropy $H$ of a discrete random variable $X$ with possible values ${x_1, x_2, ..., x_n}$ and probability mass function $P(X)$ as:
    <p>

    $$
    H(X) = E[I(X)] = E[-\ln(P(X))]
    $$

    </p>

    Here $E$ is the *expected value operator*, and $I$ is the *information content* of $X$.<br>

    It can be explicitly be written as,
    <p>

    $$
    H(X) = \sum_{i=1}^{n}P(x_i)I(x_i) = -\sum_{i=1}^{n}P(x_i)\log_b P(x_i)
    $$
    where $b$ is the base of the logarithm used. Common values of $b$ are 2, Euler's number $e$, and 10. In machine learning and deep learning, people often use $e$.

    </p>

- KL divergence from $\hat{Y}$ to $Y$ is the difference between cross entropy and entropy
<p>

$$
\mathrm{KL}(Y\|\hat{Y}) = \sum_{i}y_i\log\frac{1}{\hat{y_i}} - \sum_{i}y_i\log\frac{1}{y_i} = \sum_{i}y_i\log\frac{y_i}{\hat{y_i}}
$$

</p>

**注：** 熵的本质是香农信息量的期望，信息量就是上面公式中的$I(X)$。
- 信息量： 对信息的度量，随机变量所代表的事件发生所带来的信息的大小。出现概率小的事件信息量多，而事件发生的概率越大，则信息量越小，即信息量的大小与事件发生的概率大小成反比。
- 熵：度量了随机变量$X$平均的信息量。
- 交叉熵：使用估计出的分布q去逼近真实分布p所需要的信息
- KL散度： 交叉熵与熵的差值

***References:***
- [A Friendly Introduction to Cross-Entropy Loss](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)
- [知乎：如何通俗的解释交叉熵与相对熵?](https://www.zhihu.com/question/41252833)

<br>

* * *

<br>

# Awesome Papers
### [Very Deep Convolutional Networks for Large-Scale Image Recognition, arXiv-2014](https://arxiv.org/abs/1409.1556)
这篇论文提出了**VGG Net**。核心思想是使用小的kernel(3 * 3)来实现深度比较深的网络。使用小的kernel的原因在于，在加深网路的同时控制网络的参数不会过多。<br>
其中，该网络在训练时使用的fully connected layer，而在测试时为了适应不同大小的图片，将全连接层的参数转化为了对应参数大小的卷积核从而实现了fully convolution layer。

### [Deep Residual Learning for Image Recognition, CVPR-2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
论文中提出了deep network出现的**degradation**问题，深度网络退化问题，即：深度网络在训练是会出现其training loss比前层网络的loss要大，并将这种网络称为**plain networks**。论文中认为该问题的出现不是因为梯度消失(vanishing gradients), 因为这些plain networks中使用**BN**从而保证了在前向传播中信号是有**non-zero variance**的，并且他们通过实验证明方向传播中的由于**BN**的作用，梯度是正常的。作者认为出现这种**plain networks**的问题在于**The deep plain nets may have exponentially low convergence rates, which impact the reducing of the training error**, 而且单纯地增加迭代次数无法解决这个问题。我的理解：这些病态网络的参数空间存在很多类似于**马鞍面**的这种情况，导致梯度值变化不大从而影响了最优化。<br>
因此，该论文提出了**Resudual Learning**:
<p>

$$
\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x} 
$$

</p>

where, $\mathcal{H}(\mathbf{x})$ is output of a few stacked layers, $\mathbf{x}$ denotes the input of the first layer of these layers. It makes the network to learn the **residual** between the input and output. And this **reidual learning** is realized by **skip connection**.<br>
**注：** 参差网络学习的是输出与输入之间的参差，就是说：输出等于在输入的$\mathbf{x}$的基础上在加上$\mathcal{F}(\mathbf{x})$。而之前的方法是学习从输入到输出的mapping: $\mathbf{x} \to \mathrm{ouput}$, 并没有参差的学习。 


### [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
CMU的多人姿态估计论文，效果很不错。核心思想为：heatmap + paf，其中heatmap为多人的关键点预测，paf为骨骼bone的方向预测。通过在图像上计算每个bone的方向（使用单位向量表示）来构建对骨骼点之间的相互关系的表示。网络的label为每个骨骼点的heatmap和每个bone的paf（数量为bone的两倍，分别描述x方向和y方向）。其数据预处理部分中，使用了matlab将含有多人的同一张图片的annotation生成为多个sample，即分为`self_joints`和`others_joints`，另外其数据增强是内嵌在caffe代码中，依次对图片做了`scale`, `rotate`, `crop and pad`和`flip`的操作。