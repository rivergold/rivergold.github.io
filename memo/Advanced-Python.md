# :fallen_leaf: New

## Python 3.8

**_References:_**

- [Python 开发者: Python 3.8 新功能大揭秘](https://mp.weixin.qq.com/s?__biz=MzA4MjEyNTA5Mw==&mid=2652569675&idx=1&sn=928715a62ad52aebd66b1076c6b7b4ff&chksm=84652801b312a1170c2beb98b25431353e5dec43c346390f2bf2ce91256c0313d69c98a15c3a&mpshare=1&scene=1&srcid=#rd)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Glossary

:star2:[Python doc: Glossary](https://docs.python.org/3.8/glossary.html#term-duck-typing)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Magic Methods

## Blogs

- [A Guide to Python's Magic Methods](https://rszalski.github.io/magicmethods/)

---

## Controlling Attribute Access

**_Ref:_** [Controlling Attribute Access](https://rszalski.github.io/magicmethods/#access)

### `__getattr__(self, name)`

当访问 class 的 object 的属性时，如果没有找到对应的属性时（内部出现`AttributeError`），会调用该函数。

**Use `__getattr__` to create dynamic compositions**: [GitHub Gist example](https://gist.github.com/whanderley/3823224)

### `__setattr__(self, name, value)`

#### Be careful with recursion called

```python
def __setattr__(self, name, value):
    self.name = value
    # since every time an attribute is assigned, __setattr__() is called, this
    # is recursion.
    # so this really means self.__setattr__('name', value). Since the method
    # keeps calling itself, the recursion goes on forever causing a crash

def __setattr__(self, name, value):
    self.__dict__[name] = value # assigning to the dict of names in the class
    # define custom behavior here
```

### `__delattr__(self, name)`

### `__getattribute__(self, name)`

当访问 class 的 object 的属性时，该函数会被无条件调用；如果没有找到对应的属性，会出内部的`AttributeError`，这时如果定义了`__getattr__`函数，则会执行。

**该函数不建议定义与使用**

**_References:_**

- [简书: Python **getattribute** vs **getattr** 浅谈](https://www.jianshu.com/p/885d59db57fc)

### `@property`, `@xxx.setter`

**_Ref:_** [segmentfault: python 学习笔记-使用@property、setter、deleter](https://segmentfault.com/a/1190000007984750)

### 问题： `@property`与`__getattr__`的区别是什么

- `@property`的作用是提供给用户 get 和 set 类的 object 属性的方法，属于面向对象编程中的封装概念

- `__getattr__`的作用是无法访问 class 的 object 的属性时，进行的操作

可以仔细看下[PyTorch 源码中的`nn.Module`](https://github.com/pytorch/pytorch/blob/c002ede1075d05ab82e1d50fcc5f94ec1e0d95a9/torch/nn/modules/module.py#L577)的实现

### `__getitem__` and `__setitem__`

TODO

### `getattr` and `setattr`

TODO

**_References:_**

- :star2:[stackvoerflow: Python Newbie: using eval() to assign a value to a self.variable](https://stackoverflow.com/questions/38154901/python-newbie-using-eval-to-assign-a-value-to-a-self-variable)

---

## Pickle Object

### `__getstate__`

### `__setstate__`

<!--  -->
<br>

---

<br>
<!--  -->

# Protocol and Duck Typing

Interface: 接口
Protocol: 协议
Duck Typing: 鸭子协议

> @rivergold: 协议是非正式的接口

**_References:_**

- :thumbsup::thumbsup::thumbsup:[Book: 协议和鸭子类型](http://blog.hszofficial.site/TutorialForPython/%E8%AF%AD%E6%B3%95%E7%AF%87/%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E6%83%AF%E7%94%A8%E6%B3%95/%E5%8D%8F%E8%AE%AE%E4%B8%8E%E6%8E%A5%E5%8F%A3%E4%B8%8E%E6%8A%BD%E8%B1%A1%E5%9F%BA%E7%B1%BB.html)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Class

## metaclass

**_Ref:_** [Python 之旅: 陌生的 metaclass](http://funhacks.net/explore-python/)

### What is metaclass, class and instance?

```shell
类是实例对象的模板，元类是类的模板
+----------+             +----------+             +----------+
|          |             |          |             |          |
|          | instance of |          | instance of |          |
| instance +------------>+  class   +------------>+ metaclass|
|          |             |          |             |          |
|          |             |          |             |          |
+----------+             +----------+             +----------+
```

### Use of metaclass

```python
class Foo(metaclass=PrefixMetaclass):
    name = 'foo'
    def bar(self):
        print('bar')
```

## Initialization and Cleanup

**_References:_**

- :thumbsup:[Python 3 Patterns, Recipes and Idioms: Initialization and Cleanup](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/InitializationAndCleanup.html)

---

## `super()`

- :thumbsup::thumbsup::thumbsup:[Blog: Python: super 没那么简单](https://mozillazg.com/2016/12/python-super-is-not-as-simple-as-you-thought.html)

---

## :triangular_flag_on_post::triangular_flag_on_post::triangular_flag_on_post:Rewrite Class `__setattr__` and `__getattr__`

> @rivergold: Use `super().__setattr__` to init, otherwise will occur `RecursionError: maximum recursion depth exceeded while calling a Python object`, because follow in loop of call `__getattr__`.

```python
from pathlib import Path
import yaml


class Config(object):
    def __init__(self, cfg=None, filename=None):
        # self._cfg = cfg.copy()          ERROR
        # self._cfg_filename = filename   ERROR
        super().__setattr__('_cfg', cfg.copy())
        super().__setattr__('_cfg_filename', filename)

    @staticmethod
    def from_file(file_path):
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            log_info = '{} not exist'.format(file_path.as_posix())
            raise IOError(log_info)

        with file_path.open('r', encoding='utf-8') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

        return Config(cfg=cfg_dict, filename=file_path.name)

    def __getitem__(self, name):
        return self._cfg.get(name)

    def __setitem__(self, name, value):
        self._cfg[name] = value

    def __getattr__(self, name):
        return self._cfg.get(name)

    def __setattr__(self, name, value):
        self._cfg[name] = value

    def __iter__(self):
        return iter(self._cfg)
```

***注: *出现以上 RecursionError 错误的情况，类似于下面的这个例子：**

```python
class BadCase(object):
    def __init__(self):
        self.a = None

    @property
    def a(self):
        return self.a # 一直在获取a属性的死循环中

    @a.setter
    def a(self, value):
        self.a = value
```

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Decorator

**我的理解**:

> 装饰器是可调用的对象，其参数是另一个函数（被装饰的函数）。装饰器可能会处理被装饰的函数，然后把他返回，或者将其替换成另一个函数或可调用对象。
> 严格来说，装饰器只是语法糖。
> 装饰器的一大特性是，能把被装饰的函数替换成其他函数。第二特性是，装饰器在加载模块时立即执行。

## How to use

**_References_**

- :thumbsup:[FOOFISH-PYTHON 之禅: 理解 Python 装饰器看这一篇就够了](https://foofish.net/python-decorator.html)
- [segmentfault: Python 装饰器使用指南](https://segmentfault.com/a/1190000010681026)
- [简书: Python - 学装饰器之前，有几个点要理解](https://www.jianshu.com/p/1369d610f8bb)

## Where to use Decorator

**_References:_**

- [stackoverflow: What are some common uses for Python decorators? [closed]](https://stackoverflow.com/questions/489720/what-are-some-common-uses-for-python-decorators)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Iterator

> **Iterator** is an object which allows a programmer to traverse through all the element of a collection, regardless of its specific implementation.

**_References:_**

- :thumbsup:[zetcode: Python iterators and generators](http://zetcode.com/lang/python/itergener/)
- [知乎: 如何更好地理解 Python 迭代器和生成器？](https://www.zhihu.com/question/20829330/answer/133606850)

---

## Iterator Protocol

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-d5f332064127fc07.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="80%">
</p>

> @rivergold: Object 有`__iter__`方法时，其是可迭代对象，即`iter(object)`会返回迭代器；如果 Object 有`__next__`方法时，其是迭代器

- `iter()`将可迭代对象转化为迭代器
- 带有`__next__`方法的对象是迭代器

**_References:_**

- :thumbsup:[Blog: 完全理解 Python 迭代对象、迭代器、生成器](https://foofish.net/iterators-vs-generators.html)

---

## Iterable

可迭代的对象

> @Fluent Python: 使用 iter 内置函数可以获取迭代器对象。如果对象实现了能返回迭代器的`__iter__`方法，那么对象就是可迭代的。序列都是可以迭代的：`__getitem__`方法，而且其参数是从零开始的索引，这种对象也可以迭代。

---

## itertools

### Combinations two lists

```shell
> a = [1, 2]
> b = [4, 5]
> [[1, 4], [2, 4], [1, 5], [2, 5]]
```

Use `itertools.product`

```python
c = itertools.product(a, b)
```

**_References:_**

- :thumbsup:[stackoverflow: combinations between two lists?](https://stackoverflow.com/a/34032549/4636081)

<!--  -->
<br>

---

<br>
<!--  -->

# Generator

> @rivergold: 生成器也是迭代器，只不过是使用了`yeild`或`yeild from`。Python 使用生成器对惰性计算(lazy evaluation)提供了支持。

## Generator Function

生成器函数

> :bulb::bulb::bulb:@Fluent Python: 只要 Python 函数的定义体中有`yeild`关键字，该函数就是生成器函数。调用生成器函数时，会返回一个生成器对象。

**_References:_**

- [Blog: Python Generator](https://lotabout.me/2017/Python-Generator/)

---

## yeild from

将生成移交给子生成器

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Concurrent

## Multi Thread

**_References:_**

- [简书: python 之 ThreadPoolExecutor](https://www.jianshu.com/p/1ed39de60cb6)

---

## Multi Process

TODO:

### Multi process will deep copy global variable by default

> @rivergold: Python 多进程默认不共享全局变量，而是会将全局变量拷贝到各个进程的内存中。如果需要共享数据，需要采用进程间的通信。

**_References:_**

- [CSDN: Python 多进程默认不能共享全局变量](https://blog.csdn.net/houyanhua1/article/details/78236514)
- [stackoverflow: Globals variables and Python multiprocessing [duplicate]](https://stackoverflow.com/a/11215750/4636081)
-

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Exception

## Get exception type

```python
try:
    # do something
except Exception as e:
    print('{}: {}'.format(type(e).__name__, e))
```

**_References:_**

- :thumbsup:[stackoverflow: python: How do I know what type of exception occurred?](https://stackoverflow.com/a/9824050/4636081)
- [Python3-cookbook: 14.7 捕获所有异常](https://python3-cookbook.readthedocs.io/zh_CN/latest/c14/p07_catching_all_exceptions.html)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Functools

## partial

Partial functions allow us to fix a certain number of arguments of a function and generate a new function.

**_References:_**

- [GeeksforGeeks: Partial Functions in Python](https://www.geeksforgeeks.org/partial-functions-python/)

<!--  -->
<br>

---

<br>
<!--  -->

# Module-`inspect`

[Python doc: inspect Inspect live objects](https://docs.python.org/zh-cn/3.7/library/inspect.html)

## Example

[Taichi: kernel-materialize](https://github.com/rivergold/taichi/blob/686a0eea32088798ba21a60f78c1255fa1f5e4f0/python/taichi/lang/kernel.py#L182)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:High Performance

## Good Blogs

- [Python Bites: 5 tips to speed up your Python code](https://pybit.es/faster-python.html)

## list comprehensions vs `map`

> @python doc: Besides the syntactic benefit of list comprehensions, they are often as fast or faster than equivalent use of map.

**_References:_**

- [Python Doc: PerformanceTips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Awesome Tricks

## Make Class like a `dict`

Can use `[]` or `get` to get attribute in the class.

**Good example: [Github open-mmlab: mmcv/mmcv/utils/config.py](https://github.com/open-mmlab/mmcv/blob/9097957434da1308739c03aff53ac147f1500c56/mmcv/utils/config.py#L142)**

---

## Set random seed

```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**_Ref:_** [Github open-mmlab/mmdetection: mmdet/apis/env.py](https://github.com/open-mmlab/mmdetection/blob/58c415a069ceab37bab76c47da72824f4181cff6/mmdet/apis/env.py#L53)

---

## Convert str into class object

```python
a = 'pow'
b = eval(a)(2, 3)
print(b)
>>> 8
```

**_Ref:_** [stackoverflow: Convert string to Python class object?](https://stackoverflow.com/a/1178089)

---

## `exec` run str as Python code

```python
str_code = 'a = 1'
exec(str_code)
print(a)
>>> 1
```

---

## Compile `.py` into `.so`

Rivergold write another [memo]() for compile Python script into dynamic libs.

### Compiler Optimization

#### Loss of precision

E.g.

```python
a = 5
b = 3
c = a / b
```

After compiler optimization, c will be `int`, this may cause bug!!!

Better change your code into:

```python
a = 5.
b = 3.
c = a / b
# or
def fun(a:float, b:float) -> float:
    return a / b
```

**_References:_**

- [知乎: Python 3 新特性：类型注解](https://zhuanlan.zhihu.com/p/37239021)

---

## `format` with name arguments

E.g.

```python
x = 1
print('{x} is {x}'.format(x=x))
# print('{} is {}'.format(x, x))
# Pylint occur: Duplicate string formatting argument 'axis', consider passing as named argument
```

**_References:_**

- [stackoverflow: format strings and named arguments in Python](https://stackoverflow.com/questions/17895835/format-strings-and-named-arguments-in-python)

---

## subprocess

### Basic use

- [ ] TBD

### [Error] Bash -o: command not found

You'd better use `""` with your params in bash command.

```python
command = 'youtube-dl -f 130 your_url -o "{}"'.format('your_save_path')
subprocess.run(command, shell=True, executable="/bin/bash")
```

**_References:_**

- [stackoverflow: -o: command not found](https://superuser.com/questions/201408/o-command-not-found/201409)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Knowledge Points

## Variable Scope

**_Ref:_** [CSDN: Python 变量作用域](https://blog.csdn.net/cc7756789w/article/details/46635383)

There are four type scope in Python:

- L (Local) 局部作用域
- E (Enclosing) 闭包函数外的函数中
- G (Global) 全局作用域
- B (Built-in) 内建作用域

Python find variable in order of `L -> E -> G -> B`.

Only `def`, `class` and `lambda` will change variable scope. Other like `if/elif/else`, `try/except`, `for/while`, `with` will not change variable scope.

---

## import

### import within package

```shell
package
    - __init__.py
    - sub1
        - __init__.py
        - a.py
    - sub2
        - __init__.py
        - b.py
    c.py
d.py
```

```python
# d.py
import package.sub1.a
from package import sub1
import package.sub1.a
from package import sub1.a # Error
```

```python
# c.py
from . import sub1
from .sub1 import a
from . import sub1.a # Error
```

:triangular_flag_on_post:**Conclusion**:

```python
# in c.py
from . import x -> ok
from . import x.xx -> error: SyntaxError: invalid syntax # 不能这样写 from x import xx.xxx
# in a.py
import ..sub2 as sub2 -> error: SyntaxError: invalid syntax # 不能这样写 import ..xx
```

**_References:_**

- [博客园: python 基础之---import 与 from...import....](https://www.cnblogs.com/ptfblog/archive/2012/07/15/2592122.html)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Anaconda

## Set `ananconda/bin` in `~/.zshrc`

Edit `~/.zshrc`:

```shell
export PATH=~/software/anaconda/bin:$PATH
```

When use a new env which is different from the base, it's better to create new env for Python. Then a new folder `${ANANCONDA}/envs/<env_name>/` will created.

When install some Python package, it will also install some third party libs which will be added into `ananconda/bin`. Because of `export PATH=~/software/anaconda/bin:$PATH`, `anaconda/bin` will takes precedence over `usr/bin`. So when you want to install a package via Anaconda and it maybe conflict with system libs, you should create a new conda env.

**E.g.**

When you want to use Python with FFMPEG, `pip install opencv-python` does not with FFMPEG. `conda install opencv` has FFMPEG, and it will install it in `anaconda/bin`, but this FFMPEG not build with X264 (--disable-libx264). So you shuold create a new env for this to avoid influencing the system env.