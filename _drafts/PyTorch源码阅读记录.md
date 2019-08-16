# `setup.py`

- `lib_path`里面有什么
- `torch._C`都有什么

`build_caffe`会执行根目录（PyTorch）下的 cmakelist

```shell
cmake -DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/torch -DCMAKE_PREFIX_PATH=/home/rivergold/software/anaconda/envs/learn-pytorch/lib/python3.7/site-packages -DPYTHON_EXECUTABLE=/home/rivergold/software/anaconda/envs/learn-pytorch/bin/python -DPYTHON_INCLUDE_DIR=/home/rivergold/software/anaconda/envs/learn-pytorch/include/python3.7m -DPYTHON_LIBRARY=/home/rivergold/software/anaconda/envs/learn-pytorch/lib/libpython3.7m.so.1.0 -DTORCH_BUILD_VERSION=1.2.0a0+8554416 -DUSE_CUDA=False -DUSE_DISTRIBUTED=True -DUSE_NUMPY=True -DUSE_SYSTEM_EIGEN_INSTALL=OFF /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch
```

这一步会调用基本上所有的 cmakelist， 包括 thirdparty
