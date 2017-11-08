# Basics

## Standard notations
- Variable: $X$ (uppercase and no bold)
- Matrix: $\mathbf{X}$ (upper-case and bold)
- Vetor: $\mathbf{x}$ (lower-case and bold)
- Element/Scalar: $x$ (lower-case and no bold)

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

</p>


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
$$l
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

# Pre-installation for Deep Learning
We need:
1. Ubuntu OS
2. Nvidia GPU and driver
3. CUDA and cudann
## Install Nvidia Driver
- Notebook computer with dual graphics
    If you want to install Nvidia driver on your notebook computer which has dual graphics, it is better to install Nvidia driver by using **Additional Drivers** in Ubuntu.
    <p align="center">
        <img src="http://ovvybawkj.bkt.clouddn.com/dl/ubuntu-addtional-driver.png" width="40%">
    </p>
- Desktop computer
    To install Nvidia driver on desktop, you need to download the specific driver for your computer from [Nvidia](https://www.geforce.com/drivers). When you install the driver, you need to do the followings:
    1. Disable `Nouveau`
        ```bash
        # Inspect if there is Nouveau
        lsmod | grep nouveau
        # Create a blacklist and set nouveau into it.
        sudo gedit /etc/modprobe.d/blacklist-nouveau.conf
        ```
        Add run the following contents into the `.conf`
        ```bash
        blacklist nouveau
        options nouveau modeset=0
        ```
        Then, regenerate the kernel initramfs
        ```bash
        sudo update-initramfs -u
        ```
    2. Install Nvidia driver
        ```bash
        # Close gui
        sudo service lightdm stop
        # Install dirver
        sudo chmod u+x <nvidia-driver.run>
        sudo ./<nvidia-driver.run>
        # Reboot
        sudo reboot
        ```

## Install CUDA Toolkit
1. Download the install package from [here](https://developer.nvidia.com/cuda-downloads). It is recommended to download the `.run` to install cuda. And during installation, when it asks whether install Nvidia driver or not, please choose `No` because you have already installed the dirver. For ubuntu 16.04, it is better to install cuda 8.0+(here we use cuda_8.0.44 and we haven't try cuda9 yet), because cuda 7.5 and lower version do not support gcc > 4.9.
2. Install
    ```
    sudo bash <cuda.run>
    ```
3. Test: `cd` into the cuda folder, which default path is `/usr/local/cuda`.
    ```bash
    cd ./samples/1_Utilities/diviceQuery
    make -j4
    cd <samples path>/bin/x86_64/linux/release/
    ./deviceQuery
    ```
    Check the output to inspect if you have installed Cuda successfully or not.
4. Add Cuda into path.
    ```bash
    # Open ~/.bashrc
    sudo gedit ~/.bashrc
    # Add
    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    ```

## Install cuDNN
1. Download cuDNN from [Nvidia website]().
2. Run
    ```bash
    tar -xzvf <cudnn.tgz>
    # Copy files into CUDA Toolkit directory.
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    ```

***Reference:***
- [Nvidia: CUDA guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Nvidia: cuDNN install guide](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
- [ubuntu 16.04降级安装gcc为4.8](http://www.cnblogs.com/in4ight/p/6626708.html)
- [Ubuntu 16.04 安装英伟达（Nvidia）显卡驱动](https://gist.github.com/dangbiao1991/7825db1d17df9231f4101f034ecd5a2b)

## Build and Install Caffe1
Caffe1 need a a series of dependent libraries: OpenCV, ProtoBuffer, Boost, GFLAGS, GLOG, BLAS, HDF5, LMDB, LevelDB and Snappy. We need all of these before we start to build caffe.
### Build OpenCV
Here the version of OpenCV we build is 3.3.0 and we use `cmake-gui` to make it convenient during build configuration.<br>
The source code of OpenCV and Opencv_contrib can be downloaded from [Github](https://github.com/opencv). Before building OpenCV, we need to install some dependent libraries,
```bash
# Dependencies
sudo apt-get install build-essential checkinstall cmake pkg-config yasm gfortran git
sudo apt-get install -y libjpeg8-dev libjasper-dev libpng12-dev libtiff5-dev \
                        libavcodec-dev libavformat-dev libswscale-dev \
                        libdc1394-22-dev libxine2-dev libv4l-dev \
                        libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
                        libqt4-dev libgtk2.0-dev libtbb-dev \
                        libatlas-base-dev \
                        libfaac-dev libmp3lame-dev libtheora-dev \
                        libvorbis-dev libxvidcore-dev
# Install Python Libraries(here we use python3)
sudo apt-get install python3-dev
# OpenCV need numpy
pip install numpy
```
Then, open cmake-gui in OpenCV folder. Here are some tips needed to be aware of:
1. Check `PYTHON` configuration
2. Better to change `CMAKE_INSTALL_PREFIX` to `/usr/local/opencv3`
3. Set `OPENCV_EXTRA_MODULES_PATH` as `<the path of opencv_contirb>/modules`

After make, make install, add opencv to `PATH`
```bash
vim ~/.bashrc
# Add followings into .bashrc
export PATH=$PATH:/usr/local/opencv3/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/opencv3/lib
```

***Reference***
- [Learn Opencv: Install OpenCV3 on Ubuntu](https://www.learnopencv.com/install-opencv3-on-ubuntu/)
- [pyimagesearch: Ubuntu 16.04: How to install OpenCV](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

### Install other dependent libraries
```
sudo apt-get install -y opencl-headers build-essential protobuf-compiler \
                        libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev \
                        libsnappy-dev \
                        libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0v5 libgoogle-glog-dev \
                        libgflags-dev liblmdb-dev git python-pip gfortran
```
**Then** use cmake-gui to make caffe1, and run
```
make -j4
make runtest
make install
```

***Errors & Solution:***
- If cmake cannot find opencv and the error is like
    ```
    Could not find module FindOpenCV.cmake or a configuration file for package OpenCV.
      Adjust CMAKE_MODULE_PATH to find FindOpenCV.cmake or set OpenCV_DIR to the
      directory containing a CMake configuration file for OpenCV.  The file will
      have one of the following names:
        OpenCVConfig.cmake
        opencv-config.cmake
    ```
    Add `set(CMAKE_PREFIX_PATH <opencv install path>/share/OpenCV)` in the `CMakeLists.txt`


***Reference:***
- [BVLC/caffe wiki: Caffe installing script for ubuntu 16.04 support Cuda 8](https://github.com/BVLC/caffe/wiki/Caffe-installing-script-for-ubuntu-16.04---support-Cuda-8)

<!-- # Ubuntu gcc from 5.x to 4.8
```bash
# Check gcc version
gcc --version
# Install gcc-4.8
sudo apt-get install gcc-4.8
# gcc version
gcc --version
# List all gcc to see if gcc-4.8 installed successfully
ls /usr/bin/gcc*
# Put gcc-4.8 into priority
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100
sudo update-alternatives --config gcc
# Check gcc version again
gcc --version
```
***Reference:***
- [Ubuntu change gcc from 5.x to 4.8](http://www.cnblogs.com/in4ight/p/6626708.html) -->
