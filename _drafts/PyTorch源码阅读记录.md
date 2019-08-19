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

在`caffe2`下的`CMakeLists.txt`会编译`ATen`，并且会调用 python 去执行`caffe2/contrib/aten/gen_op.py`(传递了`aten`的路径)
作用去生成头文件？

同时，在`caffe2`下的`CMakeLists.txt`会调用 python 去执行`tools/setup_helpers/generate_code.py`，生成`autograd`所需要的代码（目测生成的是 C++代码，和 python bind 没关系的代码）
之后编译出了`torch`(C++的)

之后再调用`add_subdirectory(../torch torch)`来编译 python bind 的 torch。

# 问题

- [ ] dispatch mechanism
- [ ] add_custom_target
