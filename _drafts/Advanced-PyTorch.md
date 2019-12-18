# :fallen_leaf:torch

## What is dimension ?

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

---

## `torch.stack` vs `torch.cat`

> @rivergold: `torch.stack`会增加 n-dimension，而`torch.cat`不会增加 n-dimension, 而会增加某个 dimensionde 上的值

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:torch.Tensor

## `tensor.contiguous()`

---

## `tensor.view()` vs `tensor.reshape()`

> @rivergold: `view()` can only work on contiguous tensor. When the tensor is not contiguous, you should use `reshape()`
> 理解：当你确定 tensor 是 contiguous 时，优先选用`view`，如果不确定时，请使用`reshape`

### [ERROR] `RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead`

**_References:_**

- :thumbsup:[stackoverflow: What's the difference between reshape and view in pytorch?](https://stackoverflow.com/a/49644300/4636081)
- [stackoverflow: PyTorch - contiguous()](https://stackoverflow.com/questions/48915810/pytorch-contiguous)

> @rivergold: `permute()` will cause tensor not contiguous anymore.

**_References:_**

- [PyTorch Forum: Call contiguous() after every permute() call?](https://discuss.pytorch.org/t/call-contiguous-after-every-permute-call/13190/2)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:torch.nn

## nn.Module.named_children() vs nn.Module.named_modules()

- [PyTorch doc: named_children()](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.named_children)
- [PyTorch doc: named_modules(memo=None, prefix='')](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.named_modules)

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

**_References:_**

- :thumbsup:[PyTorch Forum: Module.children() vs Module.modules()](https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4?u=rivergold)

---

## Loss

> @rivergold loss 是`nn.Module`类型。所以如果要自定义 loss，需要继承至`nn.Module`

<!--  -->
<br>

---

<br>
<!--  -->

# torch.hub

[PyTorch doc: TORCH.HUB](https://pytorch.org/docs/stable/hub.html)

Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility.

## PyTorch Hub

[PyTorch Hub](https://pytorch.org/hub/)

<!--  -->
<br>

---

<br>
<!--  -->

# Tricks

## :bulb::bulb::bulb:InternalLayerGetter

> @rivergold: 使用`nn.ModuleDict`和`nn.Module.named_children()`从`in_model`中构建可以返回中间层输出结果的`Module`

```python
from collections import OrderedDict
import torch
import torch.nn as nn


class InternalLayerGetter(nn.ModuleDict):
    def __init__(self, in_model, internal_layer_names):
        # Inspect
        if not set(internal_layer_names.keys()).issubset(
            [name for name, _ in in_model.named_children()]):
            raise ValueError('Internal_layer_names not in model')

        self.internal_layer_names = internal_layer_names.copy()
        tmp_internal_layer_names = internal_layer_names.copy()
        modules = OrderedDict()
        for name, module in in_model.named_children():
            modules[name] = module
            if name in internal_layer_names.keys():
                tmp_internal_layer_names.pop(name)
            if not tmp_internal_layer_names:
                break
        super().__init__(modules)

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.internal_layer_names.keys():
                out[self.internal_layer_names[name]] = x
        return out
```
