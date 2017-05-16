# Theory

# Practices
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
