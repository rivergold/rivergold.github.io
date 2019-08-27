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

`caffe2/CMakeLists.txt`的主要编译工作：

基本顺序是：`caffe2/CMakeLists.txt`调用`cmake/Codegen.cmake`生成 target`ATEN_CPU_FILES_GEN_LIB`和`ATEN_CUDA_FILES_GEN_LIB` -> `caffe2/CMakeLists.txt`调用`aten/CMakeLists.txt` -> `aten/ATen/CMakeLists`会 link`ATEN_CPU_FILES_GEN_LIB`和`ATEN_CUDA_FILES_GEN_LIB`

- `INTERN_BUILD_ATEN_OPS`默认是 True
- `cmake/Codegen.cmake`根据`aten/src/ATen/Declarations.cwrap`生成的`aten/src/ATen/Declarations.yaml`很重要，跟`ATen`相关的编译基本都用到了 yaml 这个文件

1. `caff2/CMakeLists.txt`会先调用`cmake/Codegen.cmake`，做一些准备工作。暂时还不了解具体的工作。
   这部分的工作是根据 template 生成`aten/ATen`的代码并编译

2. 在`caffe2/CMakeLists.txt`会在编译`aten`时，会调用 python 去执行`caffe2/contrib/aten/gen_op.py`(传递了`aten`的路径)。这一步的`gen_op.py`使用了由`cmake/Codegen.cmake`调用`aten/src/ATen/gen.py`生成的`Declarations.yaml`。`aten/src/ATen/gen.py`在生成`Declarations.yaml`使用了`aten/src/ATen/native/native_functions.yaml`

   **\*References:**

   - [PyTorch Forum: How are python bindings created?](https://discuss.pytorch.org/t/how-are-python-bindings-created/46453/2?u=rivergold)

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
- [x] add_custom_target
- [ ] 侵入式指针
- [ ] weak references

# 零散的理解

PyTorch 的对于 Tensor operations 的代码主要为`c10`, `ATen`和原始`torch`的`TH*`的代码

**_References:_**

- [PyTorch Forum: The pytorch blog “A Tour of PyTorch Internals” is out-of-date. How to know more about the pytorch internal](https://discuss.pytorch.org/t/the-pytorch-blog-a-tour-of-pytorch-internals-is-out-of-date-how-to-know-more-about-the-pytorch-internal/19677/4?u=rivergold)

- [知乎专栏 - Gemfield: PyTorch ATen 代码的动态生成](https://zhuanlan.zhihu.com/p/55966063)

---

目录结构

**_References:_**

- [Oldpan 博客: Pytorch 源码编译简明指南](https://oldpan.me/archives/pytorch-build-simple-instruction)

---

`torch._C`是 Python `import torch`的最重要的一部分，该 module 的底层 C++代码是在`torch/src/Module.cpp`中定义的

```c++
#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", methods.data()));
#else
  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     nullptr,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
```

**_References:_**

- :thumbsup:[PyTorch Forum: Where does `torch._C` come from?](https://discuss.pytorch.org/t/where-does-torch-c-come-from/2015/3?u=rivergold)

---

`caffe2/CMakeLists.txt`中编译出的`libtorch.so`是纯 C++的 torch 库，之后会调用`add_subdirectory(../torch torch)`运行`torch/CMakeLists.txt`来编译 torch 的 Python binding 的 C++库，名叫`libtorch_python`。`libtorch.so`是`libtorch_python`的依赖库

C10 -> aten -> Caffe2 -> torch -> torch_python

---

C10，ATen 的关系

**_References:_**

- [知乎专栏 - Gemfield: PyTorch 的库文件](https://zhuanlan.zhihu.com/p/57437931)

---

PyTorch Tensor

Python `class Tensor(torch._C._TensorBase)`继承`_TensorBase`, `_TensorBase` binding C++的`THPVariable`，`THPVariable`封装的`torch::autograd::Variable`, `torch::autograd::Variable`继承的`at::Tensor`，`at::Tensor`定义在`aten/ATen/Core/Tensor.h`

我的理解：`at::Tensor`是个接口类，基于 PIml 思想，其具体实现由`TensorImpl`实现。`TensorImpl`定义在`c10/core/TensorImpl.h`中。`TensorImpl`继承`c10::intrusive_ptr_target`

`c10::intrusive_ptr_target`是一种侵入式指针，即其引用计数是放在其指向的 object 的 member 的

**_References:_**

- [知乎专栏 - Gemfield: PyTorch 的 Tensor（上）](https://zhuanlan.zhihu.com/p/54896021)
- [知乎专栏 - Gemfield: PyTorch 的 Tensor（中）](https://zhuanlan.zhihu.com/p/64135058)
- [知乎专栏 - Gemfield: PyTorch 的 Tensor（下）](https://zhuanlan.zhihu.com/p/69530008)

<!--  -->
<br>

---

<br>
<!--  -->

# 一些 C++的基础库

## gflags

Google 开源的用于命令行参数处理的库

## glog

Google 开源的用于实现应用级别的 logging 的库

## NUMA

Non-uniform memory access

一种为多处理器的计算机设计的内存架构

**_References:_**

- [Wiki: 非均匀访存模型](https://zh.wikipedia.org/wiki/%E9%9D%9E%E5%9D%87%E5%8C%80%E8%AE%BF%E5%AD%98%E6%A8%A1%E5%9E%8B)

# 边编译边了解

## `aten/src/ATen/gen.py`

```shell
python ../aten/src/ATen/gen.py --source-path ../aten/src/ATen --install_dir /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen ../aten/src/ATen/Declarations.cwrap ../aten/src/THNN/generic/THNN.h ../aten/src/THCUNN/generic/THCUNN.h ../aten/src/ATen/nn.yaml ../aten/src/ATen/native/native_functions.yaml  --output-dependencies /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen/generated_cpp.txt
```

## `gen_op.py`

```shell
python /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/caffe2/contrib/aten/gen_op.py --aten_root=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/aten --template_dir=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/caffe2/contrib/aten --yaml_dir=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen --install_dir=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/contrib/aten --aten_root=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/aten
```
