# PyTorch

# Key Concept

## Tensor

A [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to) is a multi-dimensional matrix containing elements of a single data type.

## `loss.backward()`

Computes `dloss/dx` for every parameter `x` which has `requires_grad=True`.

***References:***

- [PyTorch: What does the backward() function do?](https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944)

## `optimizer.step()`

Update parameters based on the *current* gradient (stored in `.grad` attribute of the parameter) and the update rule.

***References:***

- [PyTorch: How are optimizer.step() and loss.backward() related?](https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/2)

<br>

***

<br>

# API

## `torch.nn`

## `torchvision`

### `torchvision.transorms`

- `transforms.ToTensor()`: Convert `numpy.ndarray` or `PIL Image` into tensor. And it will convert range `[0, 255]` into `[0, 1]`

    ***References:***
    - [PyTorch Doc: torchvision.transforms.ToTensor](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor)
    - [Jonathan Hui Blog: “PyTorch - Data loading, preprocess, display and torchvision.”](https://jhui.github.io/2018/02/09/PyTorch-Data-loading-preprocess_torchvision/)

<br>

***

<br>

# Tips

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

## Only update the parameters of a few nodes

There may have 3 ways to update a part of parameters:

- set `lr=0`
- set `gradient=0`
- `optimizer.step() not update the parameter`
    - [PyTorch: Updating the parameters of a few nodes in a pre-trained network during training](https://discuss.pytorch.org/t/updating-the-parameters-of-a-few-nodes-in-a-pre-trained-network-during-training/1221)

`requires_grad`: If there’s a single input to an operation that requires gradient, its output will also require gradient.

***References:***

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
I would suggest zeroing the relevant gradients manually after calling loss.backward(). That won’t affect the gradients passed to the lower levels because the backward pass will already have been completed.

***References:***

- [PyTorch: How can I update only a part of one weight in the backpropagation](https://discuss.pytorch.org/t/how-can-i-update-only-a-part-of-one-weight-in-the-backpropagation/15229)
- [PyTorch: How to modify the gradient manually?](https://discuss.pytorch.org/t/how-to-modify-the-gradient-manually/7483)

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

***References:***

- [Blog: Pytorch中retain_graph参数的作用](https://oldpan.me/archives/pytorch-retain_graph-work)

<br>

***

<br>

# visdom

## Basics

## Plotting

### `vis.line(X, Y, opt, win)`

- `Y`: `N` or `N * M` tensor that specifies the values of the `M` lines (that connect `N` points)
- `win`: The window id. If you want to plot in the same window in different epoch.
    **Note:** If you don't set `win`, `visdom` will built a new window every time when you plot.

## Problems & Solutions

### Could not connect to Visdom server

Maybe the data use pass to `vis.line` or other plot/show function is not right. Place check your data.

<br>

***

<br>

# Common Problems & Solutions

### Error about `RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'other'`

- **Be careful about your tensor and module type is `float` or `double`**
    Some operations in PyTorch only support `float` not support `double` type. (**not sure**)
    - Use `<tensor>.float()` or `<tensor>.double()` to set tensor and module type.

    ***References:***
    - [PyTorch: Problems with weight array of FloatTensor type in loss function](https://discuss.pytorch.org/t/problems-with-weight-array-of-floattensor-type-in-loss-function/381)

- **Be careful about your tensor and module is on CPU or GPU**
    - Use `<tensor>.cuda()` or `<tensor>.cpu()` to set tensor and module on what device.

    ***References:***
    - [PyTorch: Global GPU Flag](https://discuss.pytorch.org/t/global-gpu-flag/17195)