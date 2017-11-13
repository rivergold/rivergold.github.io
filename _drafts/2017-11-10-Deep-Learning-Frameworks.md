# PyTorch
## Functions
### torch
- `torch.unsqueeze(input, dim, out=None)`
- `torch.transpose(input, dim0, dim1, out=None)` -> `Tensor`

<br>

## PyTorch Using Tricks
### Load part of pre-trained model to your model
```python
# Assume your model is `model`
pre_trained_model = torch.load(<pre-trained-model.pth>)
# Set params
layer_name = [name for name in pre_trained_model.keys()]
count = 0
for param in model.parameters():
    param.data = pre_trained_model[layer_name[count]]
    # Do not update this param
    param.requires_grad = False
    count += 1
    if count >= <lay_num>:
        break
```
***Reference:***
- [PyTorch Forums: Fcn using pretrained vgg16 in model zoo? ](https://discuss.pytorch.org/t/fcn-using-pretrained-vgg16-in-model-zoo/941)

### Convolutional layer padding
| kernel_size | stride | padding |
|:-----------:|:------:|:-------:|
|    3 * 3    |    1   |    1    |
|    5 * 5    |    1   |    2    |
|    7 * 7    |    1   |    3    |

### Set using gpu device number and id
Run your script like
```
CUDA_VISIBLE_DEVICES=<gpu_id> python <your .py script>
```
For example,
```
# One gpu
CUDA_VISIBLE_DEVICES=1 python <your .py script>
# Parallel on multiple gpus
CUDA_VISIBLE_DEVICES=2,3 python <your .py script>
```
And you can get how many gpu you can use in your code by using
```
print(torch.cuda.device_count())
```
***References:***
- [PyTorch Forums: How to specify GPU usage?](https://discuss.pytorch.org/t/how-to-specify-gpu-usage/945)

### Parallel in GPU
```
# Assume your model is a class `Net`
model = Net()
model = torch.nn.DataParallel(model)
model.cuda()
```
***References:***
- [Github: pytorch/examples/imageent/main.py](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L82)
- [PyTorch Forums: How to parallel in GPU when finetuning](https://discuss.pytorch.org/t/how-to-parallel-in-gpu-when-finetuning/796)

<br>

## Promble & Solutions
### `KeyError: 'unexpected key "module.conv1_1.weight" in state_dict'`
When load trained model, if this error occur it mean that the loading model is trained and saved by `nn.DataParallel`, but you are trying to load it without `DataParallel`.<br>
**Solution:** Here are two solutions:
- Loading model using `nn.DataParallel`
- Load the weights file, create a new ordered dict without the `module` prefix
    ```python
    # Original saved file with DataParallel
    state_dict = torch.load('myfile.pth.tar')
    # Create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # Load params
    model.load_state_dict(new_state_dict)
    ```

***Reference:***
- [PyTorch Forums: [solved] KeyError: ‘unexpected key “module.encoder.embedding.weight" in state_dict’](https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686)

<br>
<br>

# Tensorflow
Tensorflow use **computational graph** to structure its computation. You should do two discrete sections:
1. Building the computatinal graph
2. Running the computational graph

A **computational graph** is a series of TensorFlow operations arranged into a graph of nodes. Data and operations in tensorflow are nodes.

Tensorflow采用计算图的方式实现张量的计算，首先需要构建计算图，之后再运行计算图来得出计算的结构。其中，变量是计算图中的一个节点，而操作也是计算图中的节点。<br>计算图实际是基于符号计算的，在构建计算图时可以不确定每个符号的具体数值是多少，之后当计算图运行时才会根据符号所表示具体的数值来计算出结果。


- [深度学习框架的比较（MXNet, Caffe, TensorFlow, Torch, Theano)](http://kylt.iteye.com/blog/2338800) 