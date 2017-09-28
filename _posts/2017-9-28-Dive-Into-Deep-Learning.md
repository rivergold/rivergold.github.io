# Basics
## Backpropagation

<p align="center">
    <img src="http://ovvybawkj.bkt.clouddn.com/dl/backpropagation_1.png" width="60%">
</p>

<p>

Here are some notations we will need later. We use $w^l_{jk}$ to denote the weight for the connection from the $k^{th}$ neuron in the $(l - 1)^{th}$ layer to the $j^{th}$ neuron in the $l^{th}$ layer. And we use $z^{l}_j$ to represent the input of the $j^{th}$ neuron in the $l^{th}$ layer, $a^l_j$ to represent the activation output in the j^{th} neuron in the $l^{th}$ layer. Similarly, $b^l_j$ represents the bias of the $j^{th}$ neuron in the $l^{th}$ layer.
<br>
**Why use this cumbersome notation?** Maybe it is better to use $j$ to refer to the inpurt neuron, and $k$ to the output neuron. Why we use vice versa? The reason is the activation output of the $j^{th}$ neuron in the $l^{th}$ layer can be expressed like,

$$
a^l_j = \sigma(\sum_k w^l_{jk}a^{l-1}_k + b^l_j)
$$

This expression can be rewritten into a matrix from as followings,
$$
\mathbf{a}^l = \sigma(\mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l)
$$
where, $\mathbf{a}^{l}$, $\mathbf{a}^{l-1}$ and $\mathbf{b}^l$ are vectores, $\mathbf{W}^l$ is a <b>weight matirx</b> for the $j^{th}$ layer, and its $j^{th}$ row and $k^{th}$ column is $w^l_{jk}$. The elements in $j^{th}$ row of $\mathbf{W}^l$ are reprent the weights of neurons in $(l-1)^{th}$ layer connecting to the $j^{th}$ neuron in $l^{th}$ layer.<br>

Then, we define the loss function $C$, here we use the following notation(mean square error, MSE) as a example,
$$
C = \frac{1}{2}\frac{1}{m}\sum_{i}^{m}\| \mathbf{y}^{(i)} - \mathbf{a}^{L}(\mathbf{x}^{(i)}) \|^2
$$
where, $L$ denotes the number of layers in the networks, $\mathbf{a}^L$ denotes the final output of the network. And the loss of a single training example is $C_{\mathbf{x}^{(i)}} = \frac{1}{2}\|\mathbf{y}^{(i) - \mathbf{a}^L}\|^2$.<br>
<b>Note:</b> Backpropagation actually compute the partial derivatives $\frac{\partial C_{x^{(i)}}}{\partial w}$ and $\frac{\partial C_{x^{(i)}}}{\partial b}$ for single trainning example. Then, we calculate $\frac{\partial C}{\partial w}$ and $\frac{\partial C}{\partial b}$ by averageing over training samples (this step is for GD or mini-bath GD). Here we suppose the training example $\mathbf{x}$ has been fixed. And in order to simplify notation, we drop the $\mathbf{x}$ subscript, writing the loss $C_\mathbf{x}^{(i)}$ as $C$.<br>

So, for each single training sample $\mathbf{x}$, the lose maybe written as,
$$
C = \frac{1}{2}\| \mathbf{y} - \mathbf{a}^L \| = \frac{1}{2}\sum_j (y_j - a^L_j)^2
$$
Here, we define $\delta^l_j$ as
$$
\delta^l_j = \frac{\partial C}{\partial z^l_j}
$$
$\delta^l_j$ shows that the input of $j^{th}$ neuron in the $l^{th}$ layer influences the extent of the network loss change (Details can be obtained from [here](http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation)).<br>

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
- [Neural Networks and Deep Learning: How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

<br>

## Batch Normalization

<p>

Assume $\mathbf{X}$ is 4d input $(N, C, H, W)$, the output of batch normalization layer is
$$
y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
where $x$ is a mini-batch of 3d input $(N, H, W)$. The $\mathrm{E}[x]$ and $\mathrm{Var}[x]$ are calculate pre-dimension over the mini-batches and $\gamma$ and $\beta$ are learnable parameter vectors of size $C$(the input size).<br>
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
If we want output $a$ to have the same variance as all of its input $x$, the variance of $w$ needs to be $\frac{1}{n}$, $\mathrm{Var}(x) = \frac{1}{n}$, it means $w \sim \mathcal{N}(0, \frac{1}{n})$.<br>

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
<br>

# Installation
## Install Cuda, cuDNN
1. Install Cuda
2. Install cuDNN
    - Download cuDNN from [NVIDA cuDNN](https://developer.nvidia.com/cudnn)
    - Decompress it and copy `include` and `lib` file into cuda ([\*ref](https://medium.com/@acrosson/installing-nvidia-cuda-cudnn-tensorflow-and-keras-69bbf33dce8a))
        ```shell
        tar -xzvf cudnn-7.0-linux-x64-v4.0-prod.tgz
        cp cuda/lib64/* /usr/local/cuda/lib64/
        cp cuda/include/cudnn.h /usr/local/cuda/include/
        ```
