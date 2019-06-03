# PyTorch Source

- [Inside 245-5D: PyTorch internals](http://blog.ezyang.com/)

    ***Ref:*** [机器学习研究会订阅号: 揭秘PyTorch内核！核心开发者亲自全景解读（47页PPT）](https://mp.weixin.qq.com/s?__biz=MzU1NTUxNTM0Mg==&mid=2247491033&idx=2&sn=9595f55c0394675dc7b1fe16ddeb8007&chksm=fbd27178cca5f86e643f47e159f967190ea7148a7d93a58a419836f472ad6e842af82ad8cce0&mpshare=1&scene=1&srcid=#rd)

## PyTorch Internals

- [PyTorch: A Tour of PyTorch Internals (Part I)](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)
- [PyTorch: PyTorch Internals Part II - The Build System](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)

### PyTorch Code Structure

- Raw C++
- C++ warpped with PyObject
- Pure Python

### How does PyTorch extend the Python interpreter to define a Tensor type that can be manipulated from Python code?

Based on *CPython*

### How does PyTorch wrap the C libraries that actually define the Tensor’s properties and methods ?

- Define `THPTensor`
- Use `generic` to generate different type Tensor
    Because of the old version `torch` is written in C, not supported Template, so using `generic` with macro. torch after 0.2 has **ATen** written in C++.

### How does PyTorch cwrap work to generate code for Tensor methods ?

Using **cwarp** based on *yaml* to warp raw C++ source as `Python Object`

***References:***

- [Blog: 使用C写Python的模块](https://www.zouyesheng.com/python-module-c.html)

### How does PyTorch’s build system take all of these components to compile and generate a workable application?

Use the source/header files genearted above to build.

### How does a simple invocation of python setup.py install do the work that allows you to call import torch and use the PyTorch library in your code?

- Setuptools
    > Setuptools is an extension to the original distutils system from the core Python library.

    The core component of Setuptools is the setup.py file which contains all the information needed to build the project.

***

## PyTorch Source Code Tree (base on 0.1.1)

Use `generic` to auto generate different type C++ source and header files.

- torch
    - lib
        Dependent raw C++ libs
    - csrc
        - Raw C++ for Python extension
        - Using `cwarp` with `yaml` to generate *python warp c++* files during buiding

## PyTorch Build Steps

1. Build dependences
    - Build raw C++ libs
    - cwarp `nn` extentions
2. Build modules (by Setuptools)
    - Build pure python modules
    - Build extentions
        1. Build extentions modules dependend libs based on source/head files generated from *cwarp* and raw C++ libs
        2. Build extentions modules with srouce/head files and dependend libs.