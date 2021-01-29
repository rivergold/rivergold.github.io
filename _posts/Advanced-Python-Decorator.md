---
title: "[Advanced Python] Decorator"
last_modified_at: 2021-01-29
categories:
  - Memo
tags:
  - Python
---

装饰器，是给函数或者类增加其他功能的一种设计模式。

## :fallen_leaf:作用域

作用域：指变量可以被感知的空间范围

### LEGB法则

- 首先搜索局部作用域（L）
- 之后是上一层嵌套结构中的def或者lambda函数的嵌套作用域（E）
- 之后是全局作用域（G）
- 最后是内置作用域（B）

### 核心要点

- Python不是所有的语句块都会产生作用域
- Module、Class、def定义时会产生作用域
- if-elif-else、for-else、while、try-except\try-finally等关键字语句并不会产生作用域

References:
- [知乎：python的闭包、装饰器和functools.wraps](https://zhuanlan.zhihu.com/p/78500405): 文章中的参考链接也很值得看
- [博客园： Python学习之变量的作用域](https://www.cnblogs.com/fireporsche/p/7813961.html)

## :fallen_leaf:装饰器

本文介绍三种常用的装饰器写法：

- 基本装饰器
- 不修改被装饰函数信息的装饰器
- 带参数的装饰器

文本已为函数添加“写入日志”功能作为装饰器的目的，给出3个样例，完整的代码请见[GitHub](https://github.com/rivergold/learn-python/blob/main/Decorator/learn_decorator.py)。

### 基本装饰器

```python
# ---------------
# 装饰器最基础的用法
# ---------------
def auto_timer1(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        RLogger.log(
            f'{func.__name__} process time: {time.time() - start_time}')
        return res

    return wrapper

@auto_timer1
def func1(x):
    return x
```

### 不修改被装饰函数信息的装饰器

```python
# -------------------------
# 不修改被装饰函数信息的装饰器
# -------------------------
def auto_timer2(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        RLogger.log(
            f'{func.__name__} process time: {time.time() - start_time}')
        return res

    return wrapper

@auto_timer2
def func2(x):
    return x
```

### 带参数的装饰器

```python
def auto_timer3(level=RLogger.INFO):
    def auto_timer3_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            RLogger.log(
                f'{func.__name__} process time: {time.time() - start_time}',
                level=level)
            return res

        return wrapper

    return auto_timer3_wrapper

@auto_timer3(level=RLogger.WARNING)
def func3(x):
    return x
```
