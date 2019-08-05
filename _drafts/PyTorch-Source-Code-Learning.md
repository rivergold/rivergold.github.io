# PyTorch

## 学习过程

1. Python part 的实现
2. C++ part 的实现
3. C++和 Python 结合的实现

---

## Blogs

### Inside 245-5D: PyTorch internals

- [Home](http://blog.ezyang.com/)

- [pdf](http://web.mit.edu/~ezyang/Public/pytorch-internals.pdf)

**_Ref:_** [机器学习研究会订阅号: 揭秘 PyTorch 内核！核心开发者亲自全景解读（47 页 PPT）](https://mp.weixin.qq.com/s?__biz=MzU1NTUxNTM0Mg==&mid=2247491033&idx=2&sn=9595f55c0394675dc7b1fe16ddeb8007&chksm=fbd27178cca5f86e643f47e159f967190ea7148a7d93a58a419836f472ad6e842af82ad8cce0&mpshare=1&scene=1&srcid=#rd)

### 知乎-Gemfield 专栏

- [详解 Pytorch 中的网络构造](https://zhuanlan.zhihu.com/p/53927068)

---

## PyTorch Internals

- [PyTorch: A Tour of PyTorch Internals (Part I)](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)
- [PyTorch: PyTorch Internals Part II - The Build System](https://pytorch.org/blog/a-tour-of-pytorch-internals-2/)

### PyTorch Code Structure

- Raw C++
- C++ warpped with PyObject
- Pure Python

### How does PyTorch extend the Python interpreter to define a Tensor type that can be manipulated from Python code?

Based on _CPython_

### How does PyTorch wrap the C libraries that actually define the Tensor’s properties and methods ?

- Define `THPTensor`
- Use `generic` to generate different type Tensor
  Because of the old version `torch` is written in C, not supported Template, so using `generic` with macro. torch after 0.2 has **ATen** written in C++.

### How does PyTorch cwrap work to generate code for Tensor methods ?

Using **cwarp** based on _yaml_ to warp raw C++ source as `Python Object`

**_References:_**

- [Blog: 使用 C 写 Python 的模块](https://www.zouyesheng.com/python-module-c.html)

### How does PyTorch’s build system take all of these components to compile and generate a workable application?

Use the source/header files genearted above to build.

### How does a simple invocation of python setup.py install do the work that allows you to call import torch and use the PyTorch library in your code?

- Setuptools

  > Setuptools is an extension to the original distutils system from the core Python library.

  The core component of Setuptools is the setup.py file which contains all the information needed to build the project.

---

## PyTorch Source Code Tree (base on 0.1.1)

Use `generic` to auto generate different type C++ source and header files.

- torch
  - lib
    Dependent raw C++ libs
  - csrc
    - Raw C++ for Python extension
    - Using `cwarp` with `yaml` to generate _python warp c++_ files during buiding

## PyTorch Build Steps

1. Build dependences
   - Build raw C++ libs
   - cwarp `nn` extentions
2. Build modules (by Setuptools)
   - Build pure python modules
   - Build extentions
     1. Build extentions modules dependend libs based on source/head files generated from _cwarp_ and raw C++ libs
     2. Build extentions modules with srouce/head files and dependend libs.
