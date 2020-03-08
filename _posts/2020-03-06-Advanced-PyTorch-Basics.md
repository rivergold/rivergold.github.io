---
title: "[Advanced PyTorch] Basics"
last_modified_at: 2020-03-06
categories:
  - Memo
tags:
  - PyTorch
  - Memo
---

Basics about PyTorch.

## :fallen_leaf:Dynamic Graph

> @rivergold: forward 时构建图，backward 时析构图

## :fallen_leaf:Dimension

> Dimension refers to the number of `axes` needed to index.

> @rivergold: Dimension 描述了这段 tensor 数据有几个 axes/dimension。这里没有任何线性代数的含义（跟线代中矩阵的维度没有什么太大关系），仅仅只是描述这段 tensor 数据有几个 axes/dimension 的，为了方便计算机对数据进行处理。因为在实际计算机内存中，tensor 就是一段连续的数据，并没有维度概念。

E.g.

```python
x = torch.randn(1, 2, 3)
print(x.size())
>>> torch.Size([1, 2, 3])
# 这里x的dimension为3，你可以赋予这段数据物理意义：1个样本有2个三维的向量表示了3D空间中的2个点
```

**_References:_**

- :thumbsup::thumbsup::thumbsup:[stackoverflow: In Python NumPy what is a dimension and axis?](https://stackoverflow.com/a/19390939/4636081)
- [Blog: NUMPY AXES EXPLAINED](https://www.sharpsightlabs.com/blog/numpy-axes-explained/)

## :fallen_leaf:torch

### Set Random Seed

[PyTorch Doc: REPRODUCIBILITY](https://pytorch.org/docs/stable/notes/randomness.html)

> @rivergold: 当代码中使用到了 random number generator 来产生随机数时，如果想要复现同样的实验结果，需要给所有的 random number generator 设置随机种子，从而使得产生的随机数可以二次复现。

**PyTorch**

```python
seed = 123
torch.manual_seed(seed)
```

**Numpy**

```python
seed = 123
np.random.seed(seed)
```

**_Ref_** :thumbsup:[PyTorch Forum: What is manual_seed?](https://discuss.pytorch.org/t/what-is-manual-seed/5939/2?u=rivergold)

### torch.stack vs torch.cat

TODO: Need better

> @rivergold: `torch.stack`会增加 n-dimension，而`torch.cat`不会增加 n-dimension, 而会增加某个 dimension 的值

## :fallen_leaf:torch.Tensor

### tensor.contiguours()

TODO

### tensor.view() vs tensor.reshape()

> @rivergold: `view()` can only work on contiguous tensor. When the tensor is not contiguous, you should use `reshape()`
> 理解：当你确定 tensor 是 contiguous 时，优先选用`view`，如果不确定时，请使用`reshape`

使用错误时可能会出现 Error：`RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead`

**_References:_**

- :thumbsup:[stackoverflow: What's the difference between reshape and view in pytorch?](https://stackoverflow.com/a/49644300/4636081)
- [stackoverflow: PyTorch - contiguous()](https://stackoverflow.com/questions/48915810/pytorch-contiguous)

> @rivergold: `permute()` will cause tensor not contiguous anymore.

**_Ref_** [PyTorch Forum: Call contiguous() after every permute() call?](https://discuss.pytorch.org/t/call-contiguous-after-every-permute-call/13190/2)

## :fallen_leaf:torch.nn

### nn.Module.named_children() vs nn.Module.named_modules()

[PyTorch Doc: named_children()](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.named_children)
[PyTorch Doc: named_modules(memo=None, prefix='')](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.named_modules)

> @rivergold: 对于多级的 Module， `nn.Module.named_children()`会返回所有的直接相连的 children modules；`nn.Module.named_modules()`会迭代返回所有的 modules。`named_modules()`的粒度比`named_children()`更细。

> :bulb:@rivergold: 可以把多级 Module 理解为图，children_modules 就是与该 module 直接相连的 module

> :bulb::bulb::bulb:@rivergold: 神经网络是图模型，其在代码的实现也是基于图。对于 PyTorch 中的 ModuleContainer： Module, Sequential, ModuleDict，实际是在代码层面上基于模块化的思想构建了**多层级图**，高层级的图包含了底层级的图。其可视化过程就是 TensorBoard 中可视化可以逐层级展开的过程

E.g.

```python
m = nn.Sequential(nn.Linear(2, 2), nn.ReLU(),
                  nn.Sequential(nn.Sigmoid(), nn.ReLU()))
# named_children()
for module in m.named_children():
    print(module)
>>> ('0', Linear(in_features=2, out_features=2, bias=True))
    ('1', ReLU())
    ('2', Sequential(
      (0): Sigmoid()
      (1): ReLU()
    ))
# named_modules()
for module in m.named_modules():
    print(module)
>>> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): ReLU()
      (2): Sequential(
        (0): Sigmoid()
        (1): ReLU()
      )
    ))
    ('0', Linear(in_features=2, out_features=2, bias=True))
    ('1', ReLU())
    ('2', Sequential(
      (0): Sigmoid()
      (1): ReLU()
    ))
    ('2.0', Sigmoid())
    ('2.1', ReLU())
```

**_Ref_** :thumbsup:[PyTorch Forum: Module.children() vs Module.modules()](https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4?u=rivergold)

## :fallen_leaf:torch.nn.functional

TODO: 说明这个 Module 存在的作用

Basic usage is as followings

```python
import torch.nn.functional as F
```

### torch.nn.functional.pad

[PyTorch doc](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad)

:triangular_flag_on_post:E.g.

```python
x = torch.randn(1, 1, 5, 5)
y = F.pad(x, (0, 3, 0, 3))
print(y.size())
>>> torch.Size([1, 1, 8, 8])
y = F.pad(x, (0, 2, 0, 3, 0, 3))
print(y.size())
>>> torch.Size([1, 4, 8, 7])
```

> :star2:@rivergold `pad size`是成对出现的，靠前的 size 描述是对靠后的 dim(dim 的值大)的 pad

## :fallen_leaf:Loss

> :star2:@rivergold loss 是`nn.Module`类型。所以如果要自定义 loss，需要继承至`nn.Module`

## :fallen_leaf:PyTorch Hub

[PyTorch Hub](https://pytorch.org/hub/)
[PyTorch Doc: TORCH.HUB](https://pytorch.org/docs/stable/hub.html)

Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility.

## :fallen_leaf:Errors & Solutions

### [PyTorch=1.2.0] ModuleNotFoundError: No module named 'past'

```shell
    from caffe2.python import workspace
  File "/root/software/anaconda/lib/python3.7/site-packages/caffe2/python/workspace.py", line 15, in <module>
    from past.builtins import basestring
ModuleNotFoundError: No module named 'past'
```

**Solution**

```shell
pip install future
```

**_Ref_** [Github nilmtk/nilmtk: ModuleNotFoundError: No module named 'past' #548](https://github.com/nilmtk/nilmtk/issues/548)
