# Basics
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
