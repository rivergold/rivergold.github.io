# Memo

## Key Code Memo

- `Kernel ast func`: [python/taichi/lang/kernel.py](https://github.com/rivergold/taichi/blob/686a0eea32088798ba21a60f78c1255fa1f5e4f0/python/taichi/lang/kernel.py#L183)

---

## Python Module Memo

### inspect

> @rivergold: inspect 的一个功能是从将代码转化为字符串

### ast

> @rivergold: 解析 source 到 AST

**_References:_**

- [PyCoder's Weekly CN: AST 模块：用 Python 修改 Python 代码](https://pycoders-weekly-chinese.readthedocs.io/en/latest/issue3/static-modification-of-python-with-python-the-ast-module.html)

- :thumbsup:[Tobias Kohn Blog: Implementing Code Transformations in Python](https://tobiaskohn.ch/index.php/2018/07/30/transformations-in-python/)

#### Transform

Build-in class: `NodeTransformer`

> @rivergold: 改变 AST 中的 Node

### astor

> @rivergold: 解析 AST 到 source

---

### compile

**_References:_**

- :thumbsup:[Programiz: Python compile()](https://www.programiz.com/python-programming/methods/built-in/compile)

---

## Convert Python AST into LLVM IR

**_References:_**

- [Github numba/llvmlite: Generate LLVM IR from CPython ast? #488](https://github.com/numba/llvmlite/issues/488)

---

## Question

### :star2:pybind11 如何处理从 Python 传递函数到 C++的？

基于 C++的`std::function`

[Taichi: taichi/python_bindings.cpp](https://github.com/rivergold/taichi/blob/4514d5834bcc05ec5ef4aeb4c4ce7a149d98970d/taichi/python_bindings.cpp#L163)
[Taichi: taichi/program.h](https://github.com/rivergold/taichi/blob/bcd573b6e4b49bb57de3b63d45ba427b393cf3c7/taichi/program.h#L147)

### C++的代码中为什么需要使用 KernelProxy 和 Kernel?

[Taichi: taichi/program.h](https://github.com/rivergold/taichi/blob/bcd573b6e4b49bb57de3b63d45ba427b393cf3c7/taichi/program.h#L129)

~~> @rivergold: 是不是因为 pybind11 无法支持对`std::function`的封装，所以用 KernelProxy 把参数为`std::function`的函数封装进入 struct，来解决这个问题？~~

### [Pybind11] `py::gil_scoped_release release;`的作用是什么？

[Taichi: taichi/python_bindings.cpp](https://github.com/rivergold/taichi/blob/4514d5834bcc05ec5ef4aeb4c4ce7a149d98970d/taichi/python_bindings.cpp#L167)

### Python 文档生成与托管

**_References:_**

- [Github taichi-dev/taichi: Documentation #15](https://github.com/taichi-dev/taichi/issues/15)
