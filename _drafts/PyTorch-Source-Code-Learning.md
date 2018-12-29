<!-- # PyTorch Source Code -->

## PyTorch Internals

- [PyTorch: A Tour of PyTorch Internals (Part I)](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)
- []()

### How does PyTorch extend the Python interpreter to define a Tensor type that can be manipulated from Python code?

Based on *CPython*

### How does PyTorch wrap the C libraries that actually define the Tensor’s properties and methods ?

- Define `THPTensor`
- Use `generic` to generate different type Tensor

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