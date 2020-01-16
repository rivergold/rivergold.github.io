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

## Convert Python AST into LLVM IR

**_References:_**

- [Github numba/llvmlite: Generate LLVM IR from CPython ast? #488](https://github.com/numba/llvmlite/issues/488)
