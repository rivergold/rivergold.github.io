# PyTorch

**PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.**

# :fallen_leaf:Key Concept

## Dynamic Graph

**Since PyTorch is a dynamic graph framework, we create a new graph on the fly at every iteration of a training loop.**

**理解:** 每次 forward 会重新建立一个图，backward 会释放；每次的图可以不一样。

**_References：_**

- [PyTorch Blogs: PyTorch, a year in....](https://pytorch.org/blog/a-year-in/#performance)

- [知乎: 如何理解 Pytorch 中的动态图计算？](https://www.zhihu.com/question/270313536/answer/354322604)

- [知乎-Gemfield 专栏： PyTorch 的动态图(上)](https://zhuanlan.zhihu.com/p/61765561)

---

## `torch.Tensor`

A [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to) is a class, which is a multi-dimensional matrix containing elements of a single data type.

### :triangular_flag_on_post:Distinguish between the use of `torch.Tensor` and `torch.tensor`

- `torch.Tensor` is a class

- `torch.tensor` is a function to crate new tensor, when you want to create new Tensor, you need to use this function.

**_References:_**

- [stackoverflow: What is the difference between torch.tensor and torch.Tensor?](https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor)

---

## `loss.backward()`

Computes `dloss/dx` for every parameter `x` which has `requires_grad=True`.

**_References:_**

- [PyTorch: What does the backward() function do?](https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944)

---

## `optimizer.step()`

Update parameters based on the _current_ gradient (stored in `.grad` attribute of the parameter) and the update rule.

**_References:_**

- [PyTorch: How are optimizer.step() and loss.backward() related?](https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/2)

---

## Tensor Comprehensions (TC)

**Tensor Comprehensions (TC) is a tool that lowers the barrier for writing high-performance code. It generates GPU code from a simple high-level language and autotunes the code for specific input sizes.**

**_References:_**

- [PyTorch Blog: Tensor Comprehensions in PyTorch](https://pytorch.org/blog/tensor-comprehensions/)

- [机器之心: 如何通过 PyTorch 上手 Tensor Comprehensions？](https://www.jiqizhixin.com/articles/2018-03-12-5)

- [知乎: 如何看待 Tensor Comprehensions？与 TVM 有何异同？](https://www.zhihu.com/question/267167829)

---

## `torch.jit`

**A just-in-time (JIT) compiler that at runtime takes your PyTorch models and rewrites them to run at production-efficiency.**

**Uses the torch.jit compiler to export your model to a Python-free environment, and improving its performance.**

- Tracing native python code

- Compiling a subset of the python language annotated into a python-free intermediate representation

### Tracing Mode

`torch.jit.trace`, is a function that records all the native PyTorch operations performed in a code region, along with the data dependencies between them. `jit` only record native PyTorch operators.

### Script Mode

With an `@script` decorator, this annotation will transform your python function directly into high-performance C++ runtime.

**_References:_**

- [PyTorch Blog: The road to 1.0: production ready PyTorch](https://pytorch.org/blog/the-road-to-1_0/#production--pain-for-researchers)

---

## TorchScript

**Based on JIT**

**TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.**

**_References:_**

- [PyTorch: TorchScript](https://pytorch.org/docs/stable/jit.html#creating-torchscript-code)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:API

## torch

### `torch.where(condition, x, y)` $\rightarrow$ `Tensor`

**_References:_**

- [PyTorch doc: torch.where](https://pytorch.org/docs/master/torch.html#torch.where)
- [PyTorch forum: About transform wing loss to pytorch](https://discuss.pytorch.org/t/about-transform-wing-loss-to-pytorch/20045)

---

## torch.Tensor

### `torch.Tensor.item()`

Use torch.Tensor.item() to get a Python number from a tensor containing a single value

Ref [PyTorch Docs 1.0.1: TORCH.TENSOR Warning](https://pytorch.org/docs/1.0.0/tensors.html)

This will solve the `IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number`.

### `torch.Tensor.view`

**_References:_**

- [python 优先的深度学习框架: pytorch 使用 view(\*args)在不改变张量数据的情况下随意改变张量的大小和形状](https://ptorch.com/news/59.html)

### `torch.Tensor.unsqueeze(dim) -> Tensor`

Returns a new tensor with a dimension of size one inserted at the specified position.

### `torch.Tensor.is_contiguous()`

:triangular_flag_on_post:**What ops make tensor uncontiguous?**

- transpose or permute
- some Select/Slice operations, especially those with stride>1, i.e. `tensor[::2]`
- expand

**_Ref:_** [PyTorch Forum: What ops make a tensor non contiguous?](https://discuss.pytorch.org/t/what-ops-make-a-tensor-non-contiguous/3676/2)

### `torch.Tensor.size()`

Get size of tensor.

```python
x = torch.randon(3, 4, 5)
x.size()
>>> torch.Size([3, 4, 5])
x.size(0)
>>> 3
```

---

## torch.nn

### Container

#### Module

Base class for all

#### Sequential

In `nn.Sequential`, the `nn.Module` stored inside are connected in a cascaded way. And `nn.Sequential` has its own `forward()` method.

**E.g.**

```python
# Method 1: build from list
# Method 2: build from OrderedDict
```

#### ModuleList

`nn.ModuleList` does not have a default `forward()` method, and `nn.Module` sotred inside are not connected. It just a container.

#### ModuleDict

It just a container, holds submodules in a dictionary.

#### :thumbsup:When to use them?

- Usually, using `nn.Module` and `nn.Sequential` is enough.

- When use `nn.ModuleList` and `nn.ModuleDict`, remember to implement `forward()` method.

**_References:_**

- [:thumbsup:PyTorch Forum: When should I use nn.ModuleList and when should I use nn.Sequential?](https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/4?u=rivergold)

## `torch.optim`

**_References:_**

- [PyTorch Docs: TORCH.OPTIM](https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate)

## 'torch.nn.Module'

### `torch.nn.Module.childre()`

### `torch.nn.Module.modules()`

### Module.children() vs Module.modules()

**_Ref:_** [PyTorch Forum: Module.children() vs Module.modules()
](https://discuss.pytorch.org/t/module-children-vs-module-modules/4551)

---

## torchvision

### `torchvision.transorms`

- `transforms.ToTensor()`: Convert `numpy.ndarray` or `PIL Image` into tensor. And it will convert range `[0, 255]` into `[0, 1]`

  **_References:_**

  - [PyTorch Doc: torchvision.transforms.ToTensor](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor)
  - [Jonathan Hui Blog: “PyTorch - Data loading, preprocess, display and torchvision.”](https://jhui.github.io/2018/02/09/PyTorch-Data-loading-preprocess_torchvision/)

---

## torch.utils.data

### `torch.utils.data.Dataset`

### Problems & Solutions

#### `torch.utils.data.Dataset` return must be `tensor`, `numbers`, `dicts` or `lists` or `numpy`

When I return `PIL.Image`, it occur error.

```python
TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.JpegImagePlugin.JpegImageFile'>
```

<!--  -->
<br>

---

<br>
<!--  -->

# Autograd

## :triangular_flag_on_post:Get intermidiate variable grad

**_References:_**

- [stackoverflow: Why does autograd not produce gradient for intermediate variables?](https://stackoverflow.com/questions/45988168/why-does-autograd-not-produce-gradient-for-intermediate-variables)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf: PyTorch C++ Extensions

- [Doc](CUSTOM C++ AND CUDA EXTENSIONS)

> C++ extensions are a mechanism we have developed to allow users (you) to create PyTorch operators defined out-of-source, i.e. separate from the PyTorch backend. This approach is different from the way native PyTorch operations are implemented. C++ extensions are intended to spare you much of the boilerplate associated with integrating an operation with PyTorch’s backend while providing you with a high degree of flexibility for your PyTorch-based projects.

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Tips

## Init variable in layers

**Example**

```python
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)
        self.input_conv.apply(weights_init('kaiming'))
```

---

## Only update the parameters of a few nodes

There may have 3 ways to update a part of parameters:

- set `lr=0`
- set `gradient=0`
- `optimizer.step() not update the parameter`
  - [PyTorch: Updating the parameters of a few nodes in a pre-trained network during training](https://discuss.pytorch.org/t/updating-the-parameters-of-a-few-nodes-in-a-pre-trained-network-during-training/1221)

`requires_grad`: If there’s a single input to an operation that requires gradient, its output will also require gradient.

**_References:_**

- [PyTorch: Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

**For example**

```python
x = torch.randn(2,2)
y = torch.randn(2,2)
z = torch.randn(2,2, requires_grad=True)
a = x + z
a.requires_grad=Flase
>>> RuntimeError: you can only change requires_grad flags of leaf variables. If you want to use a computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().
```

> You can’t set requires_grad=False on just part of a Variable.
> I would suggest zeroing the relevant gradients manually after calling loss.backward(). That won’t affect the gradients passed to the lower levels because the backward pass will already have been completed.

**_References:_**

- [PyTorch: How can I update only a part of one weight in the backpropagation](https://discuss.pytorch.org/t/how-can-i-update-only-a-part-of-one-weight-in-the-backpropagation/15229)
- [PyTorch: How to modify the gradient manually?](https://discuss.pytorch.org/t/how-to-modify-the-gradient-manually/7483)

---

## Why and when use `loss.backward(retain_graph=True)`

Suppose you build a network with two loss, and they share a part of layers. When you backward one loss, if you set `retain_graph=Flase`, PyTorch will free all tensor calculated during forward, and then another loss cannot backward.

`retain_graph=True` can be used during GAN traning. When the batch is to train `G`, you need to set `d_loss.backward(retain_graph=True)`. Here is a example code from [Github: WonwoongCho/Generative-Inpainting-pytorch](https://github.com/WonwoongCho/Generative-Inpainting-pytorch/blob/master/run.py#L233).

```python
def backprop(self, D=True, G=True):
    if D:
        self.d_optimizer.zero_grad()
        self.loss['d_loss'].backward(retain_graph=G)
        self.d_optimizer.step()
    if G:
        self.g_optimizer.zero_grad()
        self.loss['g_loss'].backward()
        self.g_optimizer.step()
```

**_References:_**

- [Blog: Pytorch 中 retain_graph 参数的作用](https://oldpan.me/archives/pytorch-retain_graph-work)

- [jdhao's blog: Computational Graphs in PyTorch](https://jdhao.github.io/2017/11/12/pytorch-computation-graph/)

---

## Get current `lr` from optimizer

```python
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
```

**_References:_**

- [stackoverflow: PyTorch - How to get learning rate during training?](https://stackoverflow.com/a/52671057/4636081)

---

## Set L1 or L2 regularization for the loss

- L2 regularization: set `weight_decay` in `optim.SGD`

- L1 regularization
  ```python
  l1_crit = nn.L1Loss(size_average=False)
  l1_reg = 0
  for param in model.parameters():
      l1_reg += l1_crit(param)
  lambda_l1 = 5e-4
  loss += lambda_l1 * l1_reg
  ```

**_References:_**

- [PyTorch forum: Simple L2 regularization?](https://discuss.pytorch.org/t/simple-l2-regularization/139/2)

---

## :triangular_flag_on_post:Run on GPU

- [PyTorch doc: TORCH.CUDA](https://pytorch.org/docs/stable/cuda.html)

### Check if GPU is available

```python
has_gpu = torch.cuda.is_available()
```

**_References:_**

- [Github: eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/47b7c912877ca69db35b8af3a38d6522681b3bb3/train.py#L44)

### Get GPU num

```python
num_gpu = torch.cuda.device_count()
```

### Move to GPU

- `nn.Module`: change device in-place
- `Tensor`: Return new `Tensor` with specific device

```python
# Method-1
# Module: modifies in-place
model.cuda()
# Tensor: need to return new
x = x.cuda()
# Method-2
model.to(torch.device('cuda:0'))
x = x.to(torch.device('cuda:0'))
```

**_References:_**

- [PyTorch doc: torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.cuda)
- [PyTorch doc: torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.cuda)
- [PyTorch doc: torch.nn.Module.to](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.cuda)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:visdom

## Basics

### `vis.line(X, Y, opt, win)`

- `Y`: `N` or `N * M` tensor that specifies the values of the `M` lines (that connect `N` points)
- `win`: The window id. If you want to plot in the same window in different epoch.
  **Note:** If you don't set `win`, `visdom` will built a new window every time when you plot.

### `vis.images(images, nrow=8, padding=2, win)`

- `images`: as a `B * C * H * W` tensor (numpy is ok, list is ok)
- `nrow`: Number of images in a row
- `padding`: Padding around the image, equal padding around all 4 sides.

**_References:_**

- [Visdom: `vis.images`](images)
- [Github: junyanz/pytorch-CycleGAN-and-pix2pix visualizer.py](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2e04baaecab76e772cf36fb9ea2e3fe68fd72ba5/util/visualizer.py#L117)

### `vis.text`

**_References:_**

- [Visdom: `vis.text`](https://github.com/facebookresearch/visdom/blob/master/README.md#vistext)
- [Github: facebookresearch/visdom how to plot multi lines text with '\n' #154](https://www.google.com.tw/search?q=visdom+text&oq=visdom+text&aqs=chrome..69i57.2433j0j1&sourceid=chrome&ie=UTF-8)

## Problems & Solutions

### Could not connect to Visdom server

Maybe the data use pass to `vis.line` or other plot/show function is not right. Place check your data.

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Common Problems & Solutions

## Error about `RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'other'`

- **Be careful about your tensor and module type is `float` or `double`**
  Some operations in PyTorch only support `float` not support `double` type. (**not sure**)

  - Use `<tensor>.float()` or `<tensor>.double()` to set tensor and module type.

  **_References:_**

  - [PyTorch: Problems with weight array of FloatTensor type in loss function](https://discuss.pytorch.org/t/problems-with-weight-array-of-floattensor-type-in-loss-function/381)

- **Be careful about your tensor and module is on CPU or GPU**

  - Use `<tensor>.cuda()` or `<tensor>.cpu()` to set tensor and module on what device.

  **_References:_**

  - [PyTorch: Global GPU Flag](https://discuss.pytorch.org/t/global-gpu-flag/17195)

### PyTorch `torch.utils.data.Dataset` return Python `list`

It is not good to return `list` in `torch.utils.data.Dataset`, better use `numpy` or `Tensor`. If you use `list` as dataset label, PyTorch take each element in `list` as a label for a sample, not whole `list` as one label.

<br>

---

<br>

# :fallen_leaf:PyTorch on Mobile

PyTorch -> ONNX -> Caffe2

(onnx-caffe2 is in caffe2 now)

**_References:_**

- [PyTorch: Pytorch model running in Android](https://discuss.pytorch.org/t/pytorch-model-running-in-android/27238/2)
- [PyTorch 1.0 Doc: TRANSFERING A MODEL FROM PYTORCH TO CAFFE2 AND MOBILE USING ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html)
- [Github onnx/tutorials:
  Importing models from ONNX to Caffe2](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCaffe2Import.ipynb)

## Make caffe2 run on android

1. Build caffe2 for android using `pytorch/scripts/build_android.sh`

   - Change `gcc` to `clang`

2. Set CMakeLists to contain **c10**, **caffe2**, **ATen** and **google** include files.

   More details please look at the [AICamera_new](https://github.com/wangnamu/AICamera_new) project.

   **Note:** The tensor value setting in this project is incorrect. The right way is from [caffe2_cpp_tutorial](https://github.com/leonardvandriel/caffe2_cpp_tutorial/blob/a6f772a4c19b293863f12494bcc3b6ac742d3961/src/caffe2/binaries/pretrained.cc#L85)

3. Link `.so`

**_References:_**

- [Gtihub pytorch/pytorch: [Caffe2] Caffe2 for Android has no include files #14353](https://github.com/pytorch/pytorch/issues/14353)
- [Gtihub wangnamu/AICamera_new](https://github.com/wangnamu/AICamera_new)
- [Github leonardvandriel/caffe2_cpp_tutorial](https://github.com/leonardvandriel/caffe2_cpp_tutorial)

## Caffe2

### Build for android

ok

### Build for Linux

failed

### Tutorial

- [Github leonardvandriel/caffe2_cpp_tutorial](https://github.com/leonardvandriel/caffe2_cpp_tutorial)

## Problems

### Speed

- [Github onnx/onnx-caffe2: onnx-caffe2 is slower? #152](https://github.com/onnx/onnx-caffe2/issues/152)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf: Awesome Tools

## TorchSnooper

Debug PyTorch code using PySnooper

- [Github zasdfgbnm/TorchSnooper](https://github.com/zasdfgbnm/TorchSnooper)

**_References:_**

- [机器学习算法与 Python 学习: PyTorch 代码调试利器: 自动 print 每行代码的 Tensor 信息](https://mp.weixin.qq.com/s?__biz=MzIxODM4MjA5MA==&mid=2247489750&idx=3&sn=6504eec0570cc30291c21f4adec2ac72&chksm=97ea32b3a09dbba51a7736e3ac0a349ffc5d9b4f07089cbb6b2d1fd0e5b9e7890d8ce0a4a7b0&mpshare=1&scene=1&srcid=#rd)

## NVIDIA apex

- [Github](https://github.com/NVIDIA/apex)

<!--  -->
<br>

---

<!--  -->

# C++ API

## 临时

### Get data value from tensor

**_References:_**

- [Github pytorch/pytorch: at::Tensor::data() is deprecated but no other way is suggested for cpp extensions #28472](https://github.com/pytorch/pytorch/issues/28472)

### torch::max(tensor)

Return `std::tuple`

### tensor.item<float>()

**_References:_**

- [PyTorch Forum: How to convert tensor entry to a float or double?](https://discuss.pytorch.org/t/how-to-convert-tensor-entry-to-a-float-or-double/45220)

### Indexing

**_References:_**

- [PyTorch Forum: Indexing using the C++ APIs](https://discuss.pytorch.org/t/indexing-using-the-c-apis/35997)

### Some Good References

- [火星寻冰日志: PyTorch C++ 使用笔记](http://ghostblog.lyq.me/pytorch-cpp/)

- [PyTorch Forum: Survey: What are you using the C++ API for?](https://discuss.pytorch.org/t/survey-what-are-you-using-the-c-api-for/55163)

### Conditional Indexing

**_References:_**

- [PyTorch Forum: [ATen] C++ API: Equivalent to Conditional Index](https://discuss.pytorch.org/t/aten-c-api-equivalent-to-conditional-index/15682)