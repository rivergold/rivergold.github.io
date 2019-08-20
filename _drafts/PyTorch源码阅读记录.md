# `setup.py`

- `lib_path`里面有什么
- `torch._C`都有什么

`build_caffe`会执行根目录（PyTorch）下的 cmakelist

```shell
cmake -DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/torch -DCMAKE_PREFIX_PATH=/home/rivergold/software/anaconda/envs/learn-pytorch/lib/python3.7/site-packages -DPYTHON_EXECUTABLE=/home/rivergold/software/anaconda/envs/learn-pytorch/bin/python -DPYTHON_INCLUDE_DIR=/home/rivergold/software/anaconda/envs/learn-pytorch/include/python3.7m -DPYTHON_LIBRARY=/home/rivergold/software/anaconda/envs/learn-pytorch/lib/libpython3.7m.so.1.0 -DTORCH_BUILD_VERSION=1.2.0a0+8554416 -DUSE_CUDA=False -DUSE_DISTRIBUTED=True -DUSE_NUMPY=True -DUSE_SYSTEM_EIGEN_INSTALL=OFF /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch
```

这一步会调用基本上所有的 cmakelist， 包括 thirdparty

根目录下的`CMakeLists.txt`会编译 `c10` 和 `caffe2`

```shell
# ---[ Main build
add_subdirectory(c10)
add_subdirectory(caffe2)
```

`caff2/CMakeLists.txt`会先调用`cmake/Codegen.cmake`，做一些准备工作。暂时还不了解具体的工作。

在`caffe2`下的`CMakeLists.txt`会编译`ATen`，并且会调用 python 去执行`caffe2/contrib/aten/gen_op.py`(传递了`aten`的路径)
作用去生成头文件？

同时，在`caffe2`下的`CMakeLists.txt`会调用 python 去执行`tools/setup_helpers/generate_code.py`，生成`autograd`所需要的代码（目测生成的是 C++代码，和 python bind 没关系的代码）

之后调用`add_library(torch ${Caffe2_CPU_SRCS})`编译出`torch`(C++的，和 Python binding 没有关系)，这一步会根据情况编译 GPU 和 CPU 版本。

最后再调用`add_subdirectory(../torch torch)`来编译 python bind 的 torch，库名叫`torch_python`。`torch/CMakeLists.txt`仅编译 Torch Python binding。

`torch/csrc`为 C++代码，一部分为纯 C++代码，一部分是 C++ 和 Python 的混合编程；其目的是为了实现 python bind to C++，`torch`除`csrc`以外的文件都和 C++无关。

## `caffe2/CMakeLists.txt`

```shell
add_custom_command(
OUTPUT
${TORCH_GENERATED_CODE}
COMMAND
"${PYTHON_EXECUTABLE}" tools/setup_helpers/generate_code.py
  --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
  --nn-path "aten/src"
......
```

这里所用到的`${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml`是由`cmake/Codegen.cmake`根据`aten/ATen/Declarations.cwarp`生成出来的。`Codegen.cmake`会调用`aten/src/ATen/gen.py`生成`Declarations.yaml`。这里需要参考[知乎专栏-Gemfield PyTorch ATen 代码的动态生成](https://zhuanlan.zhihu.com/p/55966063)

```shell
cd cmake/
python ../aten/src/ATen/gen.py --source-path ../aten/src/ATen --install_dir /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen ../aten/src/ATen/Declarations.cwrap ../aten/src/THNN/generic/THNN.h ../aten/src/THCUNN/generic/THCUNN.h ../aten/src/ATen/nn.yaml ../aten/src/ATen/native/native_functions.yaml  --output-dependencies /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen/generated_cpp.txt
```

# 问题

- [ ] dispatch mechanism
- [ ] add_custom_target
