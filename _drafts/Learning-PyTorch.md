# PyTorch

# :fallen_leaf:Key Concept

## `torch.Tensor`

A [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to) is a multi-dimensional matrix containing elements of a single data type.

## `loss.backward()`

Computes `dloss/dx` for every parameter `x` which has `requires_grad=True`.

**_References:_**

- [PyTorch: What does the backward() function do?](https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944)

## `optimizer.step()`

Update parameters based on the _current_ gradient (stored in `.grad` attribute of the parameter) and the update rule.

**_References:_**

- [PyTorch: How are optimizer.step() and loss.backward() related?](https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/2)

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

---

## torch.nn

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

## Train on GPU

```python
# Model
model.cuda()
# Tensor
x = x.cuda()
```

Ref [PyTorch doc: torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.cuda) and [PyTorch doc: torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.cuda).

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

<!--  -->
<br>

---

<!--  -->
