# Pre-installation for Deep Learning
We need:
1. Ubuntu OS
2. Nvidia GPU and driver
3. CUDA and cudnn
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

<br>

* * *

<br>

# PyTorch
## Functions
### torch
- `torch.unsqueeze(input, dim, out=None)`
- `torch.transpose(input, dim0, dim1, out=None)` -> `Tensor`

## torch.nn
- `torch.nn.Sequential(*args)`
    A sequential container. Modules will be added to it in the order they are passed in the constructor.

## Loss Layer
### `nn.MSELoss(size_average=True, reduce=True)`
Creates a criterion that measures the mean squared error between $n$ elements in the input $x$ and target $y$:
<p>

$$
loss(x, y) = \frac{1}{n} \sum\|x_i - y_i\|^2
$$

</p>

- **size_average:** By default `size_average=True`, it calculate the mean value of loss between intput and output. If set `size_average=False` the loss is the sum of each element of input and target.
- **reduce:** By default `reduce=True` the loss calculate mode depends on **size_average**. If set `reduce=False`, this function will ignore `size_average` and return each sample's element loss in the mini-batch.
```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=0):
        super().__init__()
        self.num_samples = num_samples
    
    def __getitem__(self, index):
        x = np.zeros((2,2))
        y = np.ones((2,2))
        return torch.from_numpy(x), torch.from_numpy(y)
    
    def __len__(self):
        return self.num_samples

train_set = Dataset(4)
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=2)

# size_average=True
criterion = nn.MSELoss()

for batch_id, (input, target) in enumerate(train_set_loader):
    input_var, target_var = Variable(input), Variable(target)
    loss = criterion(input_var, target_var)
    print(loss)
    # Just run one batch and print once
    break

>>> Variable containing:
>>> 1
>>> [torch.DoubleTensor of size 1]

# size_average=False
criterion = nn.MSELoss(size_average=False)
>>> Variable containing:
>>> 8
>>> [torch.DoubleTensor of size 1]

# reduce=False
criterion = nn.MSELoss(reduce=False)
>>> Variable containing:
# 1th sample in mini-batch
>>> (0 ,.,.) = 
>>>   1  1
>>>   1  1
# 2th sample in mini-batch
>>> (1 ,.,.) = 
>>>   1  1
>>>   1  1
>>> [torch.DoubleTensor of size 2x2x2]
```
**注：** 这里的`reduce`可理解为“归纳”。另外需要注意的是，PyTorch的`MSELoss`与Caffe的`EuclideanLossLayer`的不同之处，`EuclideanLossLayer`计算
<p>

$$
loss = \frac{1}{2N}\sum_{n=1}^{N}\|\hat{y_n} - y_n\|^2
$$

</p>

`EuclideanLossLayer`取了mini-batch大小的平均，而`MSELoss`要么是计算element-size平均（所有数值都算进去），要么是求所有的和，而不会计算mini-batch的平均。

<br>

## PyTorch Using Tricks
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

**Difference between DataParallel and DistributedDataParallel**
- `DataParallel` is for perfoming training on multiple GPUs, single machine.
- `DistributedDataParallel` is for multiple machines.

***References:***
- [PyTorch Forums: What is the difference between DataParallel and DistributedDataParallel?](https://discuss.pytorch.org/t/what-is-the-difference-between-dataparallel-and-distributeddataparallel/6108)

**对PyTorch的分布式训练的浅显理解：** 单机多GPU中，将模型拷贝到不同的GPU中，每个GPU计算当前mini-bath中一部分，之后将计算的梯度的**sum/average**（单机多GPU中PyTorch是将梯度加起来的）传递到original模型里进行BP。在多机分布计算时，将输入的mini-batch分配到各个node中，之后在BP阶段将不同的node的梯度取平均之后更新参数。

### Load pre-trained model
A recommended way to save trained model is 
```python
torch.save(<model>.state_dict(), <save path>)
```
This way only saves the parameters of the trained model as a `OrderedDict`, without the structure. So when you want to load the pre-trained saved by this way, first you need to creat a object of the model, and then
```python
model_params = torch.load(<pre_trained_model path>)
# Assume your model is `model`
model.load_state_dict(model_params)
```
If you want to **load a part of a model**, for example just load previous $n$ layers of a pre-trained model,
```python
# Assume your model is `model`
trained_model_params = torch.load(<pre-trained-model.pth>)
# Set params
layer_name = [name for name in trained_model_params.keys()]
for i, param in enumerate(model.parameters()):
    param.data = pre_trained_model[layer_name[count]]
    # Do not update this param
    param.requires_grad = False
    if i > n:
        break
```
**Note:** In PyTorch, saved model state_dict save the model parameters in the order of the layer creates in the `nn.Model`.<br>

If you model is trained with `nn.DataParallel`, when you save the mode as `OrderedDict` using `torch.save(<model>.state_dict, <path>)`, it will add a string `module.` before each key. And when you load this model without `DataParallel` next time, it will occur `keyError: unexpected key "module.<xxx>" in state_dict'`, one solution is to create a new orderdict dict without `module` prefix
```python
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
```
***Reference:***
- [PyTorch Forums: Fcn using pretrained vgg16 in model zoo? ](https://discuss.pytorch.org/t/fcn-using-pretrained-vgg16-in-model-zoo/941)
- [PyTorch Forums: [solved] KeyError: ‘unexpected key “module.encoder.embedding.weight” in state_dict’](https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686)


### Train network with multiple branches
Here is an example
```python
class mm(nn.Module):
    def __init__(self):
        super(mm, self).__init__()
        self.n = nn.Linear(4,3)
        self.m = nn.Linear(3,2)
        self.m2 = nn.Linear(3,4)
    def forward(self, input, input2):
        input_ = self.n(input)
        input2_ = self.n(input2)
        o1 = self.m(input_)
        o2 = self.m2(input2_)
        return o1, o2
```
Calculate loss with PyTorch
```python
# One way
o1, o2 = mm(input)
o = o1 + o2
# Another way
l1 = loss(o1, target)
l2 = loss2(o2, target2)
torch.autograd.backward([l1, l2])
```

***References:***
- [PyTorch Forums: How to train the network with multiple branches](https://discuss.pytorch.org/t/how-to-train-the-network-with-multiple-branches/2152)


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

* * *

<br>

# Tensorflow
## Basics
### Understanding Tensorflow framework
Tensorflow use **computational graph** to structure its computation. You should do two discrete sections:
1. Building the computatinal graph
2. Running the computational graph

A **computational graph** is a series of TensorFlow operations arranged into a graph of nodes. Data and operations in tensorflow are nodes.

Tensorflow采用计算图的方式实现张量的计算，首先需要构建计算图，之后再运行计算图来得出计算的结构。其中，变量是计算图中的一个节点，而操作也是计算图中的节点。<br>计算图实际是基于符号计算的，在构建计算图时可以不确定每个符号的具体数值是多少，之后当计算图运行时才会根据符号所表示具体的数值来计算出结果。

The role of the Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run. 

***References:***
- [深度学习框架的比较（MXNet, Caffe, TensorFlow, Torch, Theano)](http://kylt.iteye.com/blog/2338800) 

## `Placeholders`
A graph can be parameterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later.
```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```

## Common Funtions
### `tf`
- `tf.argmax`: Gives the index of the highest entry in a tensor along some axis

<br>

* * *

<br>

# Caffe1
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

<br>

- When make caffe with `cudnn`, and run `make runtest` occur error like `Check failed: status == CUDNN_STATUS_SUCCESS (6 vs. 0) CUDNN_STATUS_ARCH_MISMATCH`
    This is beacuse `cudnn` requisites GPU's compute capability 3.0 or higher. If your GPU doesn't conform this requist, you should have to make caffe without cudnn

    ***References:***
    - [Google Maile-Caffe Users: Check failed: status == CUDNN_STATUS_SUCCESS (6 vs. 0) CUDNN_STATUS_ARCH_MISMATCH](https://groups.google.com/forum/#!topic/caffe-users/AEh4cqkvAIM)
    - [NVIDI: Deep Learning SDK Documentation](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

<br>

## Convolution Implement in Caffe
Caffe use a `im2col` to implement convolution, the key thought is:

<p align="center">
<tr>
<td> <img src="http://ovvybawkj.bkt.clouddn.com/caffe/caffe-convolution-1.png" alt="Drawing" style="width: 320px;"/> </td>
<td> <img src="http://ovvybawkj.bkt.clouddn.com/caffe/caffe-convolution-2.png" alt="Drawing" style="width: 320px;"/> </td>
</tr>
</p>

<p align="center">
<tr>
<td> <img src="http://ovvybawkj.bkt.clouddn.com/caffe/caffe-convolution-3.png" alt="Drawing" style="width: 320px;"/> </td>
<td> <img src="http://ovvybawkj.bkt.clouddn.com/caffe/caffe-convolution-4.png" alt="Drawing" style="width: 320px;"/> </td>
</tr>
</p>

***References:***
- [Blog: RTFSC|Caffe源码阅读（其一）](http://www.jianshu.com/p/810d30e5ebf8)
- [知乎：在Caffe中如何计算卷积](https://www.zhihu.com/question/28385679)
- [Github Yangqing/caffe: Convolution in Caffe: a memo](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo)


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

