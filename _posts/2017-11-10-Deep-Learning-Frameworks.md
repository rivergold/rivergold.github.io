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

***References:***
- [深度学习框架的比较（MXNet, Caffe, TensorFlow, Torch, Theano)](http://kylt.iteye.com/blog/2338800) 

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

