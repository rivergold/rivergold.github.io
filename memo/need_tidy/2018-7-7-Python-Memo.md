Python is a language which is flexible and easy to learn. And it has lots of convenient modules and packages.

In this article, I summarized some useful modules and packages I have used for mackine learning and data science. Hope it will give you some help or guidance:smiley:.

<!--  -->

<br>

---

<!--  -->

# Valuable Websites

- [python3-cookbook](http://python3-cookbook.readthedocs.io/zh_CN/latest/index.html)
- [Python Tips](http://book.pythontips.com/en/latest/index.html)
- [Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

<!--  -->

<br>

---

<!--  -->

# Python Build-in

## Variables

Variable in Python is a label bound to a object, it means that variable is a reference of the object.<br>
If you are a C++ programmer, you can think variable in Python as a pointer in C++ which point to the object.<br>

**理解：** Python 中的变量是对 object 的标注，也可以说是 object 的引用，而且这种引用是可以动态改变的。而 C++中的 Varible 就是那个 object, 而且 Python 中的 reference 与 C++的 reference 是有不同的： C++的 reference 就代表了那个 object，且不能改变，而 Python 中的 reference(variable)可以改变其 reference 的 object，是动态的。而且如果你是一个 C++掌握者，Python 的 Variable 更像是 C++的 pointer。

```python
a = [1, 2, 3]
b = a # b is a reference of object [1, 2, 3]
pritn(b)
>>> [1, 2, 3]
b = [1, 2] # now b is a reference of another object: [1, 2]
print(a)
>>> [1, 2, 3]
```

<!--  -->
<br>

---

<!--  -->

## Function parameters as references

Type of passing parameters to function:

- **call by value:** the function gets a copy of the argument
- **call by reference:** the function gets a pointer to the argument
- **call by sharing:** the function gets a copy of the reference of the argument, it means that the parameters inside the function become a **alias** of the arguments.

In Python, the function gets a copy of the arguments, but the arguments are always references.<br>
**理解：** Python 的函数传参，函数获得的是实参 object 的引用的拷贝<br>

**Note:** Please read **Fluent Python** book.

<!--  -->
<br>

---

<!--  -->

## Introspection

> In computer programming, `introspection` refers to the ability to examine someting to determine what it is, what it knows and what it is capcable of doing.
> 在计算机编程中，自省是指这种能力：检查某些事物以确定它是什么、它知道什么以及它能做什么。

**_References:_**

- [IBM developerWorks: Guide to Python introspection](https://www.ibm.com/developerworks/library/l-pyint/index.html)

### Frequently-used introspection function

- `getattr()`: Return the value of the named attribute of an object. If not found, it returns the default value provided to the function.

  It can be used to get dynamic attribute in Python

  **_References:_**

  - [Programiz: Python getattr()](https://www.programiz.com/python-programming/methods/built-in/getattr)
  - [stackoverflow: getting dynamic attribute in python](https://stackoverflow.com/questions/13595690/getting-dynamic-attribute-in-python)

<!--  -->
<br>

---

<!--  -->

## Operators

### `and` vs `bool`

- `and`: Tests both expression are logically `True`

- `&`: Used with `True / False` values, to test if both are `True`

**理解:** and 是判断逻辑与，而&是对两个变量进行的位操作（需要对两个）

Ref [stackoverflow: Difference between 'and' (boolean) vs. '&' (bitwise) in python. Why difference in behavior with lists vs numpy arrays?](https://stackoverflow.com/a/22647006/4636081)

由于 numpy 中的 array 没有 logical value, 所以在 numpy 中进行以下操作时，需要使用`&`

```python
a = np.random.randn(5, 5, 3)
a[(a[:,:,0]>0) & (a[:,:,1]>0)].shape
# a[np.logical_and(a[:,:,0]>0, a[:,:,1]>0)] is better
>>> (3, 3)
```

<!--  -->
<br>

---

<!--  -->

## list

- `zip(*iterables)`: Make an iterator that aggregates elements from each of the iterables.

  ```python
  x1 = [1, 2, 3]
  x2 = [4, 5, 6]
  zipped = zip(x1, x2)
  list(zipped)
  >>> [(1, 4), (2, 5), (3, 6)]
  ```

  `zip()` in conjunction with the `*` operator can be used to unzip a list:

  ```python
  y = [(1, 4), (2, 5), (3, 6)]
  x1, x2 = zip(*y)
  x1
  >>> (1, 2, 3)
  x2
  >>> (4, 5, 6)
  ```

  **_Reference:_**

  - [Python doc: 2. Built-in Functions: zip](https://docs.python.org/3/library/functions.html#zip)

- `list.insert(index, element)`: Insert element into list

  ```python
  a = [1,2,3]
  a.insert(2, 0)
  print(a)
  >>> [1,2,0,3]
  ```

- Check list is empty

  ```python
  a = []
  if not a:
      print('List is empty')
  ```

  **_References:_**

  - [stackoverflow: How do I check if a list is empty?](https://stackoverflow.com/questions/53513/how-do-i-check-if-a-list-is-empty)

- `list.append()`

  shadow copy or deep copy ?
  **`append` is shadow copy**

  **_References:_**

  - [RUNOOB.COM: Python append() 与深拷贝、浅拷贝](http://www.runoob.com/w3cnote/python-append-deepcopy.html)

### `zip`

- Combine two list:

  ```python
  a = [1, 2, 3]
  b = [4, 5, 6]
  c = list(zip(a, b))
  print(c)
  >>> [(1, 4), (2, 5), (3, 6)]
  ```

  **理解:** `zip`就是将多个 list 对应位置上的 element 分别组合起来

- Unzip a list of list/tuple

  ```python
  a = [1, 2, 3], [4, 5, 6], [7, 8, 9]
  a1, a2, a3 = zip(*a)
  print(a1, a3, a3)
  >>> (1, 4, 7) (2, 5, 8), (3, 6, 9)
  ```

  **Note:** If you regard `a` as a matrix, `zip(*a)` will get each column of `a`.

  **_References:_**

  - [stackoverflow: How to unzip a list of tuples into individual lists? [duplicate]](https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists/12974504)
  - [stackoverflow: What does the Star operator mean? [duplicate]](https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean)

<!--  -->
<br>

---

<!--  -->

## dict

- `dict1.update(dict2)`: Adds dictionary `dict2's` key-values pairs in to `dict1`. This function does not return anything.

- `dict.pop(<key>)`: Delate a key of a dict

<!--  -->
<br>

---

<!--  -->

## string

- `"` in str

  ```python
  a = '\"abc\"'
  print(a)
  >>> "abc"
  # format
  a = '"{}"'.format(123)
  print(a)
  >>> "123"
  ```

- `str.replace`

  ```python
  <str>.replace(<old_str>, <new_str>)
  ```

  You can use it to delete character by replace `<old_str>` as `''`

  **_References:_**

  - [stackoverflow: How to delete a character from a string using Python](https://stackoverflow.com/questions/3559559/how-to-delete-a-character-from-a-string-using-python)

- Remove punctuation(标点符号) in a string using `str.translate(table)`

  ```python
  import string
  trans_table = str.maketrans('', '', str.punctuation)
  data = 'this is a test!!!'
  data.translate(trans_table)
  print(data)
  >>> this is a test
  ```

  **_References:_**

  - [Python3.6 doc str.translate](https://docs.python.org/3/library/stdtypes.html?highlight=maketrans#str.translate)

### `str.rstrip()`

Remove letter to the right of the string

```python
a = '123  '
print(a.rstrip())
>>> 123
```

### `r` before string

`r"""xxx"""` represents this is a **raw string**, which tells compiler do not do transferred meaning for this string.

**理解:** `r`表示该字符串是 raw string，不需要进行转义。

**_Ref:_** [CSND: Python 字符串前面加'r'](https://blog.csdn.net/orzlzro/article/details/6645909)

Some function or class docstring will use raw string, E.g. [PyTorch source code](https://github.com/pytorch/pytorch/blob/c002ede1075d05ab82e1d50fcc5f94ec1e0d95a9/torch/nn/modules/module.py#L33).

<!--  -->
<br>

---

<!--  -->

## `queue`

```python
import queue
q = queue.Queue(10)
for data in pre_frames.queue:
    print(data)
```

**_References:_**

- [stackoverflow: Can I get an item from a PriorityQueue without removing it yet?](https://stackoverflow.com/a/43960452/4636081)

<!--  -->
<br>

---

<!--  -->

## File IO

- Copy file

  ```python
  import shutil
  shutil.copy2(<from_path>, <to_path>)
  ```

  E.G

  ```python
  shutil.copy2('./test.jpg', '/data/test_dist.jpg')
  ```

  **_References:_**

  - [stackoverflow: How do I copy a file in python?](https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python)

- Move file

  ```python
  import os
  import shutil
  # Method 1:
  os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
  # Method 2:
  shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
  ```

  **_References:_**

  - [stackoverflow: How to move a file in Python](https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python)

- Delete file

  ```python
  os.remove() # removes a file.

  os.rmdir() # removes an empty directory.

  shutil.rmtree() # deletes a directory and all its contents.

  pathlib.Path.unlink() # removes the file or symbolic link.

  pathlib.Path.rmdir() # removes the empty directory.
  ```

  **_References:_**

  - [stackoverflow: Delete a file or folder](https://stackoverflow.com/a/6996628/4636081)

<!--  -->
<br>

---

<!--  -->

## datetime

- :thumbsup:[strftime format](http://strftime.org/): Explain what `%M`, `%s` means

### Convert string into time

```python
from datetime import datetime

datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
```

**_References:_**

- :thumbsup::thumbsup:[stackoverflow: Converting string into datetime](https://stackoverflow.com/questions/466345/converting-string-into-datetime)
- [python3-cookbook: 3.15 字符串转换为日期](https://python3-cookbook.readthedocs.io/zh_CN/latest/c03/p15_convert_strings_into_datetimes.html)

### Calculate the difference between two datetime objects

```python
start_time = datetime.strptime('1:20', '%M:%S')
end_time = datetime.strptime('2:10', '%M:%S')
diff_time = (end_time - start_time).total_seconds()
print(diff_time)
>>> 50.0
```

**_References:_**

- [stackoverflow: Python - Calculate the difference between two datetime.time objects](https://stackoverflow.com/questions/43305577/python-calculate-the-difference-between-two-datetime-time-objects/43308104)

<!--  -->
<br>

---

<!--  -->

## Counter

```python
data = '12:45:56'
num_each_char = Counter(data)
print(num_each_char[':'])
>>> 3
```

**_References:_**

- [stackoverflow: How do I find the duplicates in a list and create another list with them?](https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them)

<!--  -->
<br>

---

<!--  -->

## Errors and Exceptions

- Raising exceptions

  The `raise` statement allows the programmer to force a specified exception to occur.

  ```python
  raise NameError('Error')
  ```

  E.G.

  ```python
  img = cv2.imread(img_path)
  if img is None :
      raise IOError('Load img failed: ', img_path)
  ```

  **_References:_**

  - [stackoverflow: Manually raising (throwing) an exception in Python](https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python)
  - [Python tutorial: 错误与异常](http://www.pythondoc.com/pythontutorial3/errors.html)
  - [RUNOOB.COM: Python 异常处理](http://www.runoob.com/python/python-exceptions.html)

<!--  -->
<br>

---

<!--  -->

## Built-in Functions

- `hasattr(object, name)`: Check to see if a object has the attribute

**_References:_**

- [Programiz: Python hasattr](https://www.programiz.com/python-programming/methods/built-in/hasattr)

<!--  -->
<br>

---

<!--  -->

## Magic Method

- [极客学院: 定制类和魔法方法](https://wiki.jikexueyuan.com/project/explore-python/Class/magic_method.html)

### `__getattr__`

会在对象中不存在这个属性时调用，作为**兜底**操作

```python
class Example(object):
    def __init__(self):
        self.name = 'example'
        self._data = {'a': 'a'}

    def __getattr__(self, name):
        return self._data[name]
```

**_Ref:_** [知乎: Python 实践 51-特殊方法**getattr**和**getattribute**](https://zhuanlan.zhihu.com/p/34594638)

### `__getitem__`

Can use `instance[attr_name]` to get attribute.

<!--  -->
<br>

---

<!--  -->

## Packages and `__init__.py`

**TBD**

**_References_**

- [Chris Yeh Blog: The Definitive Guide to Python import Statements](https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html)
- [python3-cookbook: 10.1 构建一个模块的层级包](https://python3-cookbook.readthedocs.io/zh_CN/latest/c10/p01_make_hierarchical_package_of_modules.html)
- [python3-cookbook: 使用相对路径名导入包中子模块](https://python3-cookbook.readthedocs.io/zh_CN/latest/c10/p03_import_submodules_by_relative_names.html)
- [stackoverflow: What's the difference between a Python module and a Python package?](https://stackoverflow.com/questions/7948494/whats-the-difference-between-a-python-module-and-a-python-package)

<!--  -->
<br>

---

<!--  -->

## Class

Member begin with `_` (e.g `_member_name`) is private member, not truly private, but for code style.

**_References:_**

- [Hackenoon: Understanding the underscore( \_ ) of Python](https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc)

- [Igorsobreira Blog: Difference between \_, ** and **xx\_\_ in Python](http://igorsobreira.com/2010/09/16/difference-between-one-underline-and-two-underlines-in-python.html)

<!-- will not be inherited into subclass. -->

### property, setter and getter

```python
class Example(object):
    def __init__(self):
        self._a = None

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        return self._a = value
```

When you write like the followings, it will occur `RecursionError: maximum recursion depth exceeded while calling a Python object` error

```python
class Example(object):
    def __init__(self):
        self.a = None

    @property
    def a(self):
        return self.a

    @a.setter
    def a(self, value):
        return self.a = value
```

Ref [stackoverflow: Using Properties in Python clases cause “maximum recursion depth exceeded” [duplicate]](https://stackoverflow.com/questions/36931415/using-properties-in-python-clases-cause-maximum-recursion-depth-exceeded)

<!--  -->
<br>

---

<!--  -->

## Python package tools

For example, `flatbuffers` use `.egg` to install into Python. When you delete the `flatbuffers-20190131040312-py3.7.egg`, it will can not be imported.

**_References:_**

- [Blog: Python 包管理工具解惑](https://blog.zengrong.net/post/2169.html)
- [stackoverflow: Differences between distribute, distutils, setuptools and distutils2?](https://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2)

<br>

---

<br>

# Modules

## pickle

1. What is pickle?

   [Offical introduction](https://docs.python.org/3/library/pickle.html) about pickle is

   > The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.

   In my view, `pickel` means _a thick cold sauce that is made from pieces of vegetables preserved in vinegar_. The implication is that you can store things for a long time. Pickel module provide you a good way to store your python data into disk, which is a persistence management. Lost of dataset is save as this way.

2) Common function:

   - Save python object into file
     `pickle.dump(object, file)`

   - Load binary data in disk into python
     `loaded_data = pickle.load(file)`

## argparse

1. What is argparse?
   _argparse_ is a parser for command-line options, arguments and sub-commands.
   [Offical introduction](https://docs.python.org/3/library/argparse.html) is

   > The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages and issues errors when users give the program invalid arguments.

2. The Basics:
   [Offical _argparse_ tutorial](https://docs.python.org/3/howto/argparse.html#id1) is a good and easy way to learn basic uses of _argparse_. Here are some tips I summarized.

   - Basic format(maybe it is a template) of using _argparse_
     ```python
     import argparse
     parser = argparse.ArgumentParser()
     parser.parse_args()
     ```
   - Here are two kinds of arguments:

     - _positional arguments_
       A positional argument is any argument that's not supplied as a `key=value` pair.
     - _optional arguments_
       A optional argument is supplied as a `key=value` pair.

   - Basic command

     ```python
     import argparse
     parser = argparse.ArgumentParser(description='echo your input')
     # positional arguments
     parser.add_argument('echo')
     # optional arguments
     parser.add_argument('-n', '--name')
     args = parser.parse_args()
     print(args.echo)
     ```

     The input this in shell and get output

     ```shell
     $ python <script_name>.py 'It is a basic use of argparse' --name 'My name is Kevin Ho'
     [output]: It is a basic use of argparse
               My name is Kevin Ho
     ```

     - Create a parser
       `parser = argparse.ArgumentParser(description=<>)`

     - Add arauments
       `ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])`

       For more details about the parameters you can browse [The add_argument() method](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument)

     - Parse arguments
       `args = parser.parse_args()`

### Tips

- `'action=store_true'`

  ```python
  parser.add_argument('--gpu', action='store_true')
  ```

  If you add `bool=True`, it will throw an error.

  **_References:_**

  - [Python Issue 24754](https://bugs.python.org/issue24754)

* parse from dict

  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument('name')
  parser.add_argument('--mode')
  opt = parser.parse_args(['a', '--mode=b'])
  ```

  **_Ref:_** [Python Doc: argparse — Parser for command-line options, arguments and sub-commands](https://docs.python.org/3/library/argparse.html#option-value-syntax)

## collections

This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.

- Reference
  - [不可不知的 Python 模块: collections](http://www.zlovezl.cn/articles/collections-in-python/)

### `defaultdict`

**_Referneces:_**

- [简书：Python 中 collections.defaultdict()使用](https://www.jianshu.com/p/26df28b3bfc8)

## codecs

This module defines base classes for standard Python codecs (encoders and decoders) and provides access to the internal Python codec registry, which manages the codec and error handling lookup process.

- Reference

  - [Offical Website](https://docs.python.org/3/library/codecs.html#standard-encodings)

  - [Standard Encodings](https://docs.python.org/3/library/codecs.html#standard-encodings)

## shutil

> The shutil module offers a number of high-level operations on files and collections of files. In particular, functions are provided which support file copying and removal.

**How to rename a folder?**

- Using `os.rename(<old folder path>, <new folder path>)`
- Using `shutil.move(<old folder path>, <new folder path>)`

**_Reference:_**

- [stackoverflow: How to rename a file using Python](https://stackoverflow.com/questions/2491222/how-to-rename-a-file-using-python)
- [stackoverflow: When using python os.rmdir, get PermissionError: [WinError 5] Access is denied](https://stackoverflow.com/questions/36360167/when-using-python-os-rmdir-get-permissionerror-winerror-5-access-is-denied)

## random

This module implements pseudo-random number generators for various distributions.

- `random.seed(a=None, version=2)`
  Initialize the random number genearator

  ```python
  for i in range(3):
      random.seed(123)
      print(random.randint(1, 10))
      >>> 1
      >>> 1
      >>> 1
  ```

  **Note:** `random.seed()` set the seed only work for once.<br>
  随机数种子 seed 只有一次有效，在下一次调用产生随机数函数前没有设置 seed，则还是产生随机数。

  **_References:_**

  - [random.seed(): What does it do?](https://stackoverflow.com/questions/22639587/random-seed-what-does-it-do)

- `random.shuffle(x)`
  Shuffle the squence x in place<br>

  ```python
  a = [1, 2, 3, 4]
  random.shuffle(a)
  >>> [2, 1, 4, 3]
  ```

- `random.sample(population, k)`
  Return a $k$ length list of unique elements chosen from the population sequences or set.
  ```python
  # Generate a random sequences 0 - 99, length: 10
  random_ids = random.sample(range(100, 10))
  ```

## Enum

Enum in python

```python
from enum import IntEnum
class Color(IntEnum):
    BLUE = 0
    GREEN = 1
    RED = 2
# get a list of this enum
color = [x for x in Color]
```

If we do not use `IntEnum`, another choice is

```python
from enum import Enum
class Color(Enum):
    BLUE = 0
    GREEN = 1
    RED = 2
# get a list of this enum
color = [x.value for x in Color]
```

**_References:_**

- [stackoverflow: How to get all values from python enum class?](https://stackoverflow.com/questions/29503339/how-to-get-all-values-from-python-enum-class)

## multiprocessing

**_References:_**

- [Blog: Python 多进程库 multiprocessing 中进程池 Pool 类的使用](https://blog.csdn.net/jinping_shi/article/details/52433867)
- [Blog: Python 多进程 multiprocessing.Pool 类详解](https://blog.csdn.net/SeeTheWorld518/article/details/49639651)

**When using `Pool` how to pass multiple arguments to function?**

```python
from multiprocessing import Pool
def func_worker(arg_1, arg_2):
    <do what you want to>

pool = Pool(4) # Using 4 core
pool.starmap(func_worker, <data_need_processed>)
```

**Note:** Each element of <data_need_processed> is a list or tuple which contain both of `arg_1` and `arg_2`

E.G.

```python
row_data = pd.read_csv('./image_row_data.csv')
def download_worker(data_1, data_2):
    image_name = data_1
    url = data_2
    url = 'http://' + url
    logging.error(image_name)
    urllib.request.urlretrieve(url, './images/' + image_name)

pool = Pool(8)
download_row_data =  list(zip(row_data.loc[:, 'img_name'], row_data.loc[:, 'url']))
pool.starmap(download_worker, download_row_data)
```

**_References:_**

- [stackoverflow: Python multiprocessing pool.map for multiple arguments](https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments)
- [Rawidn's Blog: Python Pool.map 多参数传递方法](https://www.rawidn.com/posts/Python-multiprocessing-for-multiple-arguments.html)
- [CODE Q&A: Python 多处理 pool.map 多个参数](https://code.i-harness.com/zh-CN/q/530d5e)

### Problems and Solutions

#### Error when using `pool.join()`

```shell
"File ***/anaconda/lib/python3.6/multiprocessing/pool.py", line 545, in join
    assert self._state in (CLOSE, TERMINATE)
AssertionError
```

The correct way to use `pool.join()` is

```python
pool.close()
pool.join()
```

**_References:_**

- [CSDN: Python 多进程 multiprocessing.Pool 类详解](https://blog.csdn.net/SeeTheWorld518/article/details/49639651)

## logging

Basic using of `logging`

```python
import logging
# Config: output file, format, level
logging.basicConfig(filename=<outputfile.log>, format='%(message)s', level=logging.DEBUG)
logging.debug(<infor>)
logging.info(<infor>)
logging.warning(<infor>)
logging.error(<infor>)
logging.critical(<infor>)
```

level: - `DEBUG` - `INFO` - `WARNING` - `ERROR` - `CRITICAL`

`CRITICAL` is the highest level, when you set level as `CRITICAL`, other level information will not be print. The default level is `WARNING`.

`format` will set the format of the logging information format. The default format is `%(levelname)s:%(name)s:%(message)s`. Other format seting can be get from [Python doc-LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes) and [简书: python3 logging 学习笔记](https://www.jianshu.com/p/4993b49b6888)

**_References:_**

- [Python doc: Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [简书: python3 logging 学习笔记](https://www.jianshu.com/p/4993b49b6888)
- [python3-cookbook: ](http://python3-cookbook-personal.readthedocs.io/zh_CN/latest/c13/p11_add_logging_to_simple_scripts.html)

**Note:** By default, when you rerun python script, `logging` will not clear the pre `.log` file, it will add information at the end of the log file. If you want to clear the pre log file,
one solution is to checking the file exist first and remove it before logging print.

```python
import os
log_file_name = <file_name.log>
if os.path.exit(log_file_name): os.remove(log_file_name)
```

**_References:_**

- [stackoverflow: How to find if directory exists in Python](https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python)
- [stackoverflow: How to delete a file or folder](https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder)

### logger and handler

Here is a e.g.

```python
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s')
log_file_handler = logging.FileHandler('./test.log')
log_file_handler.setFormatter(formatter)
log_stream_handler = logging.StreamHandler()
log_stream_handler.setFormatter(formatter)
logger.addHandler(log_file_handler)
logger.addHandler(log_stream_handler)
for i in range(10):
    logger.info(str(i))
```

**_References:_**

- [简书: python logging 模块使用教程](https://www.jianshu.com/p/feb86c06c4f4)
- [Blog: Python 标准模块 logging](https://blog.csdn.net/fxjtoday/article/details/6307285)

**Close logger:**

```python
handlers = self.log.handlers[:]
for handler in handlers:
    handler.close()
    self.log.removeHandler(handler)
```

**_References:_** [stackoverflow: python does not release filehandles to logfile](https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile)

## string

- `string.ascii_letters`: String of all lowercase and uppercase letter. `'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'`

- `string.digtis`: String of all digits `'string.digits'`

Generate random string

```python
import string
import random
random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(<string length>)])
```

## json

- `json.dumps`: `dict` to `string`
- `json.load`: `string` to `dict`

### Save json with utf-8

**_References:_**

- [stackoverflow: Saving utf-8 texts in json.dumps as UTF8, not as \u escape sequence](https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence)

## pathlib

`from pathlib import Path`

### `Path.glob(<path>)`

Glob the given pattern in the directory represented by this path, yielding all matching files (of any kind)

> The “\*\*” pattern means “this directory and all subdirectories, recursively”. In other words, it enables recursive globbing.

### `Path.rglob(<path>)`

This is like calling Path.glob() with “\*\*” added in front of the given pattern

**_References:_**

- [stackoverflow: Recursive sub folder search and return files in a list python](https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python)
- [Python doc: pathlib.Path.rglob](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob)

### `Path.as_posix()`

Return a string representation of the path with forward slashes (/)

```python
import cv2
# p is img_path
img = cv2.imread(p.as_posix())
```

**_References:_**

- [Python doc: PurePath.as_posix()](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.as_posix)

## subprocess

### Basic Use

```python
import subprocess
cmd = <your command> # E.g. 'ls'
subprocess.run(cmd, shell=True)
```

- `shell=True`: If your `command` is written as `'ls -l'`, you need to set `shell=True`; if `command` is written as `['ls', '-l']`, you don't need to set `shell=False`.

### :triangular_flag_on_post: Synchronous and asynchronous run

`subprocess.run()` will wait for command to complete, so it is synchronous.

**But**, if you write like this

```python
cmd = <your command>
p = subprocess.Popen(cmd, shell=True)
```

This is asynchronous !

E.g.

```python
cmd = 'echo subprocess'
subprocess.run(cmd, shell=True)
print('ok)
>>> subprocess
>>> ok
# If you want to wait
# p.wait()
```

```python
cmd = 'echo subprocess'
p = subprocess.Popen(cmd, shell=True)
print('ok)
>>> ok
>>> subprocess
```

**_References:_**

- [博客园: Python 多进程（1）——subprocess 与 Popen()](https://www.cnblogs.com/security-darren/p/4733368.html)

- [stackoverflow: Why is subprocess.Popen not waiting until the child process terminates?](https://stackoverflow.com/questions/1541273/why-is-subprocess-popen-not-waiting-until-the-child-process-terminates?rq=1)

---

### Run bash command and get output

```python
import subprocess
cmd = <your command> # E.g. 'ls'
retval = subprocess.check_out(cmd, shell=True)
print(retval.decode('utf-8))
```

**_Reference:_**

- [python3-cookbook: 13.6 执行外部命令并获取它的输出](https://python3-cookbook.readthedocs.io/zh_CN/latest/c13/p06_executing_external_command_and_get_its_output.html)

- [stackoverflow: Store output of subprocess.Popen call in a string](https://stackoverflow.com/questions/2502833/store-output-of-subprocess-popen-call-in-a-string)

---

<!-- Run bash command in Python.

```python
import subprocess
p = subprocess.Popen(<command>, shell=True, stdout=subprocess.PIPE)
info, error = p.communicate()
info = info.decode('utf-8')
```

- `stdout=subprocess.PIPE`: Use `PIPE` to get stdout.

**_References:_**

- [stackoverflow: Actual meaning of 'shell=True' in subprocess](https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess)
- [Blog: python 中的 subprocess.Popen（）使用](https://www.cnblogs.com/zhoug2020/p/5079407.html)
- [Linux 运维笔记: Python subprocess 模块](https://blog.linuxeye.cn/375.html)

Run bash command with input

```python
p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, executable='/bin/zsh')
p.communicate(input='yes\n'.encode())
```

**_References:_**

- [stackoverflow: Python - How do I pass a string into subprocess.Popen (using the stdin argument)?](https://stackoverflow.com/a/165662/4636081) -->

## concurrent

**_References:_**

- [掘金: Python 广为使用的并发处理库 futures 使用入门与内部原理](https://juejin.im/post/5b1e36476fb9a01e4a6e02e4)

## ast

Abstract syntax tree.

### `ast.literal_eval(str)`

```python
print(ast.literal_eval('(12, 12, 12)'))
>>> (12, 12, 12)
```

**_References:_**

- [stackoverflow: Convert a String representation of a Dictionary to a dictionary?](https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary)
- [Python Doc: ast.literal_eval](https://docs.python.org/3/library/ast.html#ast.literal_eval)

## `configparser`

Load config file.

**_Ref:_** [python3-cookbook: 13.10 读取配置文件](https://python3-cookbook.readthedocs.io/zh_CN/latest/c13/p10_read_configuration_files.html)

## `os`

### `os.path`

[Python Doc: os.path — Common pathname manipulations](https://docs.python.org/3/library/os.path.html)

#### `os.path.expanduser`

Expand path with `~/`

**_Ref:_** [Blog: [Python] 使用 os.path.expanduser() 展開含有 "~" 的路徑](https://ephrain.net/python-%E4%BD%BF%E7%94%A8-os-path-expanduser-%E5%B1%95%E9%96%8B%E5%90%AB%E6%9C%89-%E7%9A%84%E8%B7%AF%E5%BE%91/)

## collections

### OrderedDict

**Init**

```python
# Method 1: use dcit
data = OrderedDict({'a': 1, 'b': 2})
# Method 2: use list of tuple
data = OrderedDict([('a', 1), ('b', 2)])
```

**_Ref:_** [PyTorch Doc: torch.nn.Sequential(\*args)](https://pytorch.org/docs/stable/nn.html#sequential)

---

## re

### `r'\n'`

> @rivergold: 字符串前面加入`r`，表明该字符串的转义字符不起作用

### `re.sub(pattern, repl, string, count=0, flags=0)`

#### `repl`: string

#### `repl`: function

**Come from:** [Github pytorch-memo](https://github.com/rivergold/pytorch/blob/cc06e2f9479b8fc09e4557eeba6a0806c66ed160/aten/src/ATen/code_template.py#L71)

```python
import re

number_mapping = {'1': 'one',
                  '2': 'two',
                  '3': 'three'}
s = "1 testing 2 3"
print(re.sub(r'\d', lambda x: number_mapping[x.group()], s))
>>> one testing two three
```

**_References:_**

- [stackoverflow: Passing a function to re.sub in Python](https://stackoverflow.com/questions/18737863/passing-a-function-to-re-sub-in-python)

### `re.DOTALL` and `re.MULTILINE`

**Come from:** [Github pytorch-memo](https://github.com/rivergold/pytorch/blob/cc06e2f9479b8fc09e4557eeba6a0806c66ed160/aten/src/ATen/code_template.py#L25)

- `re.MULTILINE`: re 会对每行进行单独匹配

**_References:_**

- :thumbsup:[Blog: Python 正则表达式里的单行 s 和多行 m 模式](https://www.lfhacks.com/tech/python-re-single-multiline)

<br>

---

<br>

# Packages

## Matplotlib

### Concept

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-234a58290805e2c8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="70%">
</p>

Ref [matplotlib tutorial: Parts of a Figure](https://matplotlib.org/tutorials/introductory/usage.html#parts-of-a-figure)

**_References:_**

- [CSDN: OpenCV GrabCut 算法 物体分割(python 语言)](https://blog.csdn.net/qq_29300341/article/details/79026392)

### Tips

- Draw heatmap

  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  a = np.random.random((24, 24))
  plt.show(a, cmap='jet', interpolation='nearest')
  ```

- Refresh picture([\*ref](https://stackoverflow.com/questions/20936817/how-do-i-redraw-an-image-using-pythons-matplotlib))

  ```python
  fig, ax = plt.subplots(figsize=(12,8))
  fig.show()
  plt.pause(0.5)
  for i in range(100):
      ax.plot(<your data>)
      fig.canvas.draw()
  ```

- Change figure window name([\*ref](https://stackoverflow.com/questions/5812960/change-figure-window-title-in-pylab))

  ```python
  fig, ax = plt.subplots(figsize=(12,8))
  fig.canvas.set_window_title(<window title>)
  ```

- Reference
  - [Stackoverflow: Plotting a 2D heatmap with Matplotlib](http://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib)
  - [Matplot: color example](https://matplotlib.org/examples/color/colormaps_reference.html)

<br>

## Numpy

### Multiplication computation functions

- `np.dot(x1, x2)`: Dot product
  Dot product of two arrays.
  For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors (without complex conjugation). For N dimensions it is a sum product over the last axis of a and the second-to-last of b:
- `np.outer(x1, x2)`: Outer product
- `np.multiply(x1, x2)`: Elementwise multiplication

**Note:** `np.dot()` performs a matrix-matrix or matrix-vector multiplication.

### Random

- `np.random.seed(seed=None)`
  Seed the generator. **Note:** The seed is only work for once.

- `np.random.randn(d0, d1, ..., dn)`:
  Return a sample from the **standard normal** distribution.

- `class np.random.RandomState(seed=None)`
  ```python
  r = np.random.RandomState(123)
  r.randn(2,2)
  ```

**_References:_**

- [SciPy.org: Numpy Random sampling](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html)

### Shape transform

- `np.squeeze(a, axis=None)`: Remove single-dimensional entries from the shape of array.

### Change element value

- `np.where(condition[, x, y])`: Return elements, either from x or y, depending on condition.

  ```python
  a = np.random.randn(2,2)
  >>> array([[ 0.04584302, -0.24067515],
             [ 0.09577436,  0.5294515 ]])
  b = np.where(a>0.5, 1, 0)
  >>> array([[0, 0],
             [0, 1]])
  ```

  **_References:_**

  - [Stackoverflow: convert numpy array to 0 or 1](https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1)

### Create array

- `np.meshgrid(x1, x2, ..., xn)`: Return coordinate matrices from coordinate vectors.

  ```python
  x = np.linspace(0, 1, nx)
  y = np.linspace(0, 1, ny)
  xx, yy = np.meshgrid(x, y)
  ```

  **Note:** This function is used often when you want to draw your prediction model's decision boundary.

  **_References:_**

  - [deeplearning.ai: Planar data classification with one hidden layer/planar_utils.py](https://github.com/XingxingHuang/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week3/Planar%20data%20classification%20with%20one%20hidden%20layer/planar_utils.py#L7)

### Copy array

```python
a = np.zeros((2,2))
b = np.copy(a)
```

**Note!** Numpy arrays work differently than lists do. When use `b = a[::]` in numpy, `b` is not a deep copy of a, but a view of a.

**_References:_**

- [stackoverflow: Numpy array assignment with copy](https://stackoverflow.com/questions/19676538/numpy-array-assignment-with-copy)

### Copy in numpy

**_References:_**

- [CSDN:【Python】numpy 中的 copy 问题详解](https://blog.csdn.net/u010099080/article/details/59111207)

### `np.newaxis` add new axis to array

```python
x = np.array([1, 2, 3, 4])
x = x[:, np.newaxis]
print(x)
>>> array([[1],
           [2],
           [3],
           [4],
           [5]])
x = np.array(range(4)).reshape(2, 2)
x = x[..., np.newaxis]
print(x.shape)
>>> (2, 2, 1)
```

### `stack`

- `stack`: stack list of np.ndarray along axis

```python
X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
```

**_References:_**

- [CSDN: Python numpy 函数 hstack() vstack() stack() dstack() vsplit() concatenate()](https://blog.csdn.net/garfielder007/article/details/51378296)

### `np.tile(A, reps)`

Construct an array by repeating `A` the number of times given by `reps`.

E.g: convert `[1, H, W]` tensor into `[3, H, W]`.

```python
img_tensor = np.tile(img_tensor, (3, 3, 1))
```

**_References:_**

- [SciPy.org numpy.tile](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.tile.html)
- [Github junyanz/pytorch-CycleGAN-and-pix2pix util.py](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2e04baaecab76e772cf36fb9ea2e3fe68fd72ba5/util/util.py#L17)

## OpenCV

Using `import cv2` to import OpenCV

### Install `opencv-python`

```shell
pip install opencv-python
```

**_References:_**

- [Pypi: opencv-python 3.4.1.15](https://pypi.org/project/opencv-python/)

### Basic functions

- `cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)`

  **_References:_**

  - [Python cv2.normalize() Examples](https://www.programcreek.com/python/example/89398/cv2.normalize)

### Rotate image

Function: `cv2.warpAffine(img, M, dsize, flags=, borderMode=, borderValue=)`<br>
We need `rotate center`, `degree` to calculate the rotate marix

```python
# Calculate rotate matrix
M = cv2.getRotationMatrix2D(rotate_center, degree)
# Rotate
img_rotated = cv2.warpAffine(img, M, img.shape[0:2])
```

**Note:** `cv2.warpAffine` need the size of rotated image. So if you want to rotated without cropped, you need to calculate the size of rotated image manually

```python
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat
```

**_References_**:

- [OpenCV doc: warpAffine](https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)
- [stackoverflow: Rotate an image without cropping in OpenCV in C++](https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c)

### Put text

- cv2.getTextSize(text, fontFace, fontScale, thickness): Return text size (w, h, baseline)

```python
font = cv2.Font_HERSHEY_SIMPLEX
random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(10)])
font_scale = np.random.uniform(0.5, 1)
thickness = random.randint(1, 3)
(fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
x = random.randint(0, max(0, w - 1 - fw))
y = random.randint(fh, h - 1 - baseline)
color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
cv2.putText(img, random_str, (x, y), font, font_scale, color, thickness)
```

**_References:_**

- [OpenCV doc: getTextSize](https://docs.opencv.org/3.3.1/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82)

### Capturing mouse click events

**_References:_**

- [pyimagesearch: Capturing mouse click events with Python and OpenCV](https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/)

### Problems & Solution

- When OpenCV read color image, color information is stored as BGR, in order to convert to RGB

  ```python
  import cv2
  import matplotlib.pyplot as plt
  img = cv2.imread(<img_path>) # BRG
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img_rgb)
  plt.show()
  ```

  **_References:_**

  - [PHYSIOPHILE: WHY OPENCV USES BGR (NOT RGB)](https://physiophile.wordpress.com/2017/01/12/why-opencv-uses-bgr-not-rgb/)

- When using OpenCV in IPython or Jupyter notebook, `cv2.imshow` and `cv2.waitKey(0)` cause crash

  Solution: Add `cv2.destroyAllWindows()`

  ```python
  cv2.imshow('img', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  **_References:_**

  - [Github Opencv/opencv Issues: opencv cv2.waitKey() do not work well with python idle or ipython](https://github.com/opencv/opencv/issues/6452)

- When use relative path like `~/image/xxx.png`, `cv2.imread()` can not read image
  **You'd better load image with absolute path!**

  **_References:_**

  - [Github opencv/opencv: Different pathname resolving functionalities of imread in C++, Python](https://github.com/opencv/opencv/issues/7219)

### Read/Write video

- Load a video

  ```python
  cap = cv2.VideoCapture("demo.mp4")
  ```

- Get video frame number

  ```python
  cap = cv2.VideoCapture("video.mp4")
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print( length )
  ```

  **_References:_**

  - [stackoverflow: How to know total number of Frame in a file with cv2 in python](https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python)

- Get sepcific frames of the video:

  ```python
  cap.set(1, <sepcific frame id>)
  ret, frame = cap.read()
  ```

  **_References:_**

  - [stackoverflow: Getting specific frames from VideoCapture opencv in python](https://stackoverflow.com/questions/33523751/getting-specific-frames-from-videocapture-opencv-in-python)
  - [stackoverflow: OpenCV-Python cv2.CV_CAP_PROP_POS_FRAMES error](https://stackoverflow.com/questions/38563079/opencv-python-cv2-cv-cap-prop-pos-frames-error)

- Write video

  ```python
  cap = cv2.VideoCapture(0)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
  while(True):
      ret, frame = cap.read()
      if ret is True:
          # Write the frame into the file 'output.avi'
          out.write(frame)
          # Display the resulting frame
          cv2.imshow('frame',frame)
          # Press Q on keyboard to stop recording
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      else:
          break
  cap.release()
  out.release()
  ```

  **_References:_**

  - [Learn OpenCV: Read, Write and Display a video using OpenCV ( C++/ Python )](https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)

- Get video frame rate

  ```python
  video = cv2.VideoCapture(0)
  fps = video.get(cv2.CAP_PROP_FPS)
  ```

  **_References:_**

  - [Learn OpenCV: How to find frame rate or frames per second (fps) in OpenCV (Python / C++)](https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/)

## Cython

**What is Cython?** The Cython language is a superset of the Python language that additionally supports calling C functions and declaring C types on variables and class attributes. This allows the compiler to generate very efficient C code from Cython code.
Cython 是包含 C 数据类型的 Python，Cython 编译器会转化 Python 代码为 C 代码，这些 C 代码均可以调用 Python/C 的 API

**_References:_**

- [Github: cython/cython](https://github.com/cython/cython)
- [Cython 0.28.2 documentation](http://docs.cython.org/en/latest/src/tutorial/)
- [Gitbook: Cython 官方文档中文版](https://moonlet.gitbooks.io/cython-document-zh_cn/content/ch1-basic_tutorial.html)

## easydict

**EasyDict** allows to access dict values as attributes(works recursively).

```python
from easydict import EasyDict as edict
d = edict({'foo':3, 'bar':{'x':1, 'y':2}})
print(d.foo)
print(d.bar.x)
```

**_Referneces:_**

- [Github: makinacorpus/easydict](https://github.com/makinacorpus/easydict)
- [Github: endernewton/tf-faster-rcnn/lib/model/config.py](https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/model/config.py)

## conda

- `conda creat -n <env name> python=<python version>`: Create a new env
- `conda env list`: List all env
- `conda env remove -n <env name>`: Remove a env
- `conda activate <env name>`: Activate a env
- `conda deactivate`: Deactivate to base

### Use `pip` with conda

When you want to use `conda env create` a new environment, you'd better use `conda env create -n <env_name> pip`. It will install pip in the virtual environment and allow you install package into the env and not influence the **base** env.

Tips: When the env has no packages you import in the script, if will use the base env's packages.

**_References:_**

- [Conda doc: Installing non-conda packages](https://conda.io/docs/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages)
- [Github ContinuumIO/anaconda-issues: Use 'pip install' in the virtual environment created by conda #1429](https://github.com/ContinuumIO/anaconda-issues/issues/1429)

## requests

### How to post a opencv image

First you need to know the `type` and `field` of your post. For example,

```python
# Post type: file
# Post field: image
url = <your url>
img = cv2.imread(<img_path>)
_, img_encoded = cv2.imencode('.jpg', img)
files = [('image', img_encoded.tostring())]
response = requests.post(url, files=files)
print(response.text)
```

**_References:_**

- [GitHubGist kylehounslow/client.py Send and receive images using Flask, Numpy and OpenCV](https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594)
- [Blog: python 中将图片从客户端(client)推到(POST)到服务器端(server)的方法](https://www.cnblogs.com/arkenstone/p/7338241.html)

## flask

### Start a flask app

**_References:_**

- [Blog: Build a Flask App in 30 Minutes](https://stormpath.com/blog/build-a-flask-app-in-30-minutes)
- [Blog: flask 应用启动](https://blog.csdn.net/sinat_36651044/article/details/77532510)

## PIL

- Load and show image

  ```python
  from PIL import Image
  img = Image.open(<img_path>)
  img.show()
  ```

**Note:** PIL save image as RGB.

- Convet between PIL and numpy

  ```python
  # Numpy to PIL
  img = Image.fromarray(<np.ndarray>)
  # PIL to Numpy
  img = np.array(<PIL.Image>)
  ```

  **_References:_**

  - [Blog: PIL 中的 Image 和 numpy 中的数组 array 相互转换](https://www.cnblogs.com/gongxijun/p/6114232.html)
  - [stackoverflow: How to convert a PIL Image into a numpy array?](https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array)

- Draw text

  **_References:_**

  - [Pillow(PIL Fork)](https://pillow.readthedocs.io/en/3.0.x/reference/ImageDraw.html#example-draw-partial-opacity-text)

  Get text size

  **_References:_**

  - [Pillow: PIL.ImageDraw.ImageDraw.textsize](https://pillow.readthedocs.io/en/5.1.x/reference/ImageDraw.html)

  Draw text stroke(文本影音)
  **_References:_**

  - [Text Stroke(文本描边 CSS 演示)](http://www.css88.com/tool/css3Preview/Text-Stroke.html)
  - [stackoverflow: Is there a way to outline text with a dark line in PIL?](https://stackoverflow.com/questions/41556771/is-there-a-way-to-outline-text-with-a-dark-line-in-pil)
  - [[Image-SIG] how to draw a stroke/outline around text with PIL?](https://mail.python.org/pipermail/image-sig/2009-May/005681.html)

## ffmpeg

- Cut video with `start time` and `end time`

  **_References:_**

  - [stackoverflow: Cutting the videos based on start and end time using ffmpeg](https://stackoverflow.com/questions/18444194/cutting-the-videos-based-on-start-and-end-time-using-ffmpeg)

- Check video information: `ffmpeg -i xxx.mp4`

  Video size information is `width * height`.
  Video `720P`, here `720` is video's height.

  **_References:_**

  - [Blog: ffmpeg 查看视频信息](https://blog.csdn.net/caiqiiqi/article/details/74695819)

## wget

```shell
pip install wget
```

```python
import wget
wget.download(<link>, out=<name>)
```

**_References:_**

- [stackoverflow: Python equivalent of a given wget command](https://stackoverflow.com/a/28313383/4636081)

## Setuptools

<br>

---

<br>

# Python Tips

## Change python packages download source.

- **pip**

  - On Windows:

    - New a folder called `pip` under path `c:/user/<your user name>/`, and new a file name `pip.ini`, write followings in it:
      ```python
      [global]
      index-url = https://mirrors.aliyun.com/pypi/simple
      trusted-host =  mirrors.aliyun.com
      ```

  - On Linux:
    - New a folder called `.pip` under `~/`, create a file named `pip.conf` and write the same as Windows in the file.<br>

- **conda**
  Input followings in terminal

  ```python
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --set show_channel_urls yes
  ```

**_Reference:_**

- [segmentfault: pip 设置阿里云的镜像源，速度超级快](https://segmentfault.com/a/1190000006111096)

## Organize your python package

Assume that your package is like,

**理解：** 通过`__init__`实现层级暴露

```bash
package
  __init__.py
  |- subpackage1
      |- __init__.py
      |- a.py
  |- subpackage2
      |- __init__.py
      |- b.py
```

Each `__init__.py` is

```python
# package's __init__py
from . import subpackage1
from . import subpackage2
```

```python
# subpackage1's __init__py
from . import a
```

```python
# subpackage2's __init__py
from . import b
```

Then, you can import package like this,

```python
from package import subpackage1
print(subpackage1.a.<variable>)
```

If your `subpackage1's __init__py` and `subpackage2's __init__py` are empty, you need

```python
import package.subpackage1 import a
print(a.<variable>)
# import package.subpackage1.a import <variable>
# print(<variable>)
```

If `b.py` want to import method or variable in `a.py`,

```python
# b.py
from ..a import <method>
```

**_Referneces:_**

- [python3-cookbook: 10.3 使用相对路径名导入包中子模块](http://python3-cookbook.readthedocs.io/zh_CN/latest/c10/p03_import_submodules_by_relative_names.html)

## Python script import from parent directory

Assume you have some scripts like

```shell
src
  |- part_1
      |- a.py
  |- part_2
      |- b.py
```

And `b.py` want to use method or variable in `a.py`, one solution is to add parent path into `sys.path`

```python
import sys
sys.append('..')
import a
```

## How print in one with dynamically refresh in Python?

- Python3

  ```python
  print(data, end='\r', flush=True)
  ```

- Python2
  ```python
  import sys
  sys.stdout.write('.')
  # or from Python 2.6 you can import the `print` function from Python3
  from __future__ import print_function
  ```

## `format` print

```python
x = 1234.56789

# Two decimal places of accuracy
format(x, '0.2f')
>>> '1234.57'

# Right justified in 10 chars, one-digit accuracy
format(x, '>10.1f')
>>> '    1234.6'

# Left justified
format(x, '<10.1f')
>>> '1234.6    '

# Centered
format(x, '^10.1f')
>>> '  1234.6  '

# Inclusion of thousands separator
format(x, ',')
>>> '1,234.56789'
format(x, '0,.1f')
>>> '1,234.6'

# string format
temp = '{:0<10.2}'.format(x)
print(temp)
>>> 1.2e+03000
temp = '{:0<10.2f}'.format(x)
>>> 1234.57000
temp = temp = '{:0<10.2E}'.format(x)
>>> 1.23E+0300
```

**_References:_**

- [python3-cookbook: 3.3 数字的格式化输出](http://python3-cookbook.readthedocs.io/zh_CN/latest/c03/p03_format_numbers_for_output.html)

Print `{` in `format`

```python
print('{{}} {}'.format('abc'))
```

**_References:_**

- [stackoverflow: How can I print literal curly-brace characters in python string and also use .format on it?](https://stackoverflow.com/questions/5466451/how-can-i-print-literal-curly-brace-characters-in-python-string-and-also-use-fo)

## `enumerate` using details

Common use is like followings,

```python
for counter, value in enumerate(some_list):
    print(counter, value)
```

`enumerate(iterable, start=0)`, optional parameters `start` decide the start number of counter,

```python
a = ['apple', 'banana', 'orange']
for counter, value in enumerate(a, 1)
    print(counter, value)
>>> 1 apple
>>> 2 banana
>>> 3 orange
```

**_References:_**

- [Python Tips: 13. Enumerate](http://book.pythontips.com/en/latest/enumerate.html)

## Create a directory/folder if it does not exist

```python
import os
if not os.path.exists(<directory_path>):
    os.makedirs(<directory_path>)
```

**_Reference:_**

- [stackoverflow: How can I create a directory if it does not exist?](https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist)

## How to run `pip` in python script?

```python
import pip
pip.main(['install', '-r'. 'requirements.txt'])
```

**_Reference:_**

- [stackoverflow: use “pip install/uninstall” inside a python script](https://stackoverflow.com/questions/12937533/use-pip-install-uninstall-inside-a-python-script)
- [stackoverflow: How to pip install packages according to requirements.txt from a local directory?](https://stackoverflow.com/questions/7225900/how-to-pip-install-packages-according-to-requirements-txt-from-a-local-directory)

## `pip` install set package version

```shell
pip install "package>=0.2,<0.3"
```

**_References:_**

- [stackoverflow: How to pip install a package with min and max version range?](https://stackoverflow.com/questions/8795617/how-to-pip-install-a-package-with-min-and-max-version-range)

## Parallel in Python

**_Reference:_**

- [知乎专栏： Python 多核并行计算](https://zhuanlan.zhihu.com/p/24311810)

## C++ Embed Python

**_Reference:_**

- [Embedding Python in C/C++: Part I](https://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I)
- [CSDN Blog: C++调用 python](http://blog.csdn.net/marising/article/details/2917892)

## Download image from url

**_References:_**

- [stackoverflow: Downloading a picture via urllib and python](https://stackoverflow.com/questions/3042757/downloading-a-picture-via-urllib-and-python)
- [stackoverflow: AttributeError: 'module' object has no attribute 'urlretrieve'](https://stackoverflow.com/questions/17960942/attributeerror-module-object-has-no-attribute-urlretrieve)

Error `ConnectionResetError: [Errno 104] Connection reset by peer`: this error means that the request is so frequently, the server refuse some request. One solution is to do delay when download, here is an example:

```python
for url in urls:
    for i in range(10):
        try:
            r = requests.get(url).content
        except Exception, e:
            if i >= 9:
                do_some_log()
            else:
                time.sleep(0.5)
        else:
            time.sleep(0.1)
            break
     save_image(r)
```

**_References:_**

- [segmentfault: Python 频繁请求问题: [Errno 104] Connection reset by peer](https://segmentfault.com/a/1190000007480913)

## Copy in Python

**_References:_**

- [Blog: Python 正确复制列表的方法](https://www.cnblogs.com/ifantastic/p/3811145.html)

## `*` and `**` in python

This [stackoverflow answer](https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters) show most use of `*` and `**`.
And you can do like this:

```python
def fun(a, b):
    print(a, b)
data = (20, 30)
fun(*data)
```

## Python extract commpressed file

**_References:_**

- [Blog: python 解压压缩包的几种方法](https://blog.csdn.net/luoshengkim/article/details/46647423)

## Python list file in recursion

Using `pathlib`

```python
from pathlib import Path
result = list(Path(<base path>).rglob('*.jpg'))
```

**_References:_**

- [stackoverflow: Recursive sub folder search and return files in a list python](https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python)

## Python read file and remove `\n`

```python
with open(<file_path>) as file:
    data = file.read().splitlines()
```

**_References:_**

- [stackoverflow: Getting rid of \n when using .readlines() [duplicate]](https://stackoverflow.com/questions/15233340/getting-rid-of-n-when-using-readlines)

## Python class method name rule

- `_name`: Private or Protected member
- `name_`: In order to distinguish with Python **key word**
- `__name__`: Python internal name

**_References:_**

- [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#3162-naming-convention)

## Covert string representation of list to list

```python
import ast
x = "['a', 'b']"
x = ast.literal_eval(x)
```

**_References:_**

- [stackoverflow: Convert string representation of list to list](https://stackoverflow.com/a/1894296/4636081)

## Convert `byte` into string

```python
a = b'abcd'
b = a.decode('utf-8')
```

**_References:_**

- [stackoverflow: Convert bytes to a string?](https://stackoverflow.com/a/606199/4636081)

## Python get caller info

**_References:_**

- [Blog: Python 如何在函数内获取调用者信息](http://blog.0x01.site/2016/02/04/Python%E5%A6%82%E4%BD%95%E5%9C%A8%E5%87%BD%E6%95%B0%E5%86%85%E8%8E%B7%E5%8F%96%E8%B0%83%E7%94%A8%E8%80%85%E4%BF%A1%E6%81%AF/)

## pip install via gitlab subdictionary

- [stackoverflow: How can I install from a git subdirectory with pip?](https://stackoverflow.com/questions/13566200/how-can-i-install-from-a-git-subdirectory-with-pip)
- [stackoverflow: pip install from git repo branch](https://stackoverflow.com/questions/20101834/pip-install-from-git-repo-branch)

## Get caller function name using `traceback` or `inspect`

If you want to get the caller function name inside another function in Python, here are two ways:

- Using `traceback`

  ```python
  import traceback

  def fun_a():
      fun_b()

  def fun_b():
      name = traceback.extract_stack()[-2][0]
      print(name)
  ```

  Ref [CSDN: Python-被调用函数中获取调用函数信息](https://blog.csdn.net/crazycui/article/details/52129716) and [stackoverflow: Print current call stack from a method in Python code](https://stackoverflow.com/questions/1156023/print-current-call-stack-from-a-method-in-python-code)

- Using `inspect`

  ```python
  import inspect

  def fun_a():
      fun_b()

  def fun_b():
      name = inspect.stack()[1][3]
      print(name)
  ```

  Ref [stackoverflow: Getting the caller function name inside another function in Python?](https://stackoverflow.com/questions/900392/getting-the-caller-function-name-inside-another-function-in-python)

## Print current properties and values of an object

```python
dir(<object>)
```

Ref [stackoverflow: Is there a built-in function to print all the current properties and values of an object?](https://stackoverflow.com/questions/192109/is-there-a-built-in-function-to-print-all-the-current-properties-and-values-of-a)

## Compile Python package into dynamic library

Ref [Jan Buchar Blog: Using Cython to protect a Python codebase](https://bucharjan.cz/blog/using-cython-to-protect-a-python-codebase.html)

- [ ] TBD

<br>

---

<br>

# Problems and Solutions:

- `UnicodeEncodeError: 'ascii' codec can't encode character '\u22f1' in position 242`
  - [解决 Python3 下打印 utf-8 字符串出现 UnicodeEncodeError 的问题](https://www.binss.me/blog/solve-problem-of-python3-raise-unicodeencodeerror-when-print-utf8-string/)

## Python 同级目录包导入问题，使用"."错误

在 package 中使用`python script`时，相对目录会失效

Ref to [曙暮之光: python 同级目录包导入问题，使用"."错误](https://www.cnblogs.com/liuda9495/p/8351978.html)

## `python -m`

Ref [python3-cookbook: 10.3 使用相对路径名导入包中子模块](https://python3-cookbook.readthedocs.io/zh_CN/latest/c10/p03_import_submodules_by_relative_names.html)

**_References:_**

- [Blog: Python 自定义包下不同目录单元测试的导入错误](http://lessisbetter.site/2016/01/08/package-unittest-import-error/)

## Long string into muilti line

```python
data = (
    'a-{} ',
    'b-{} ',
    'c-{} '
).format(1, 2, ,3)
```

**_References:_**

- [stackoverflow: Pythonic way to create a long multi-line string](https://stackoverflow.com/questions/10660435/pythonic-way-to-create-a-long-multi-line-string)

<br>

---

<br>

# Python Performance Analysis

Here are

Time:

- Time function with decrorator
- Build-in `timeit` module
- Unix time command `/usr/bin/time`

Memory:

- `memory_profiler`
- `heapy`

## Time

### Unix time command `/usr/bin/time`

```shell
/usr/bin/time -p python <python script>

# Output
real         0.02   # Total time
user         0.01   # CPU time without kernel
sys          0.00   # Kernel function time
```

## Memory

### `memory_profiler`

1. Install `conda install memory_profiler`

2. Edit your code with `@profile`

3. Run `python -m memory_profiler <your script>`

You can use `mprof` to draw image of memory use:

1. `mprof run <your script>`
2. `mprof plot`

**_References:_**

- [PYPI: memory-profiler 0.55.0](https://pypi.org/project/memory-profiler/)

#### Problems & Solutions

##### Error `libc++abi.dylib: terminating with uncaught exception of type NSException`

This error is caused by matplotlib, you need to change matplotlib backend.

**_Solution:_**

- [Github palantir/python-language-server: libc++abi.dylib: terminating with uncaught exception of type NSException | macOS #217](https://github.com/palantir/python-language-server/issues/217#issuecomment-364967166)

<br>

---

<br>

# Python Custom Sort Key Function

```python
def intra_frame_filter_person_bboxes(person_bboxes, scores=None):
    """[summary]

    Arguments:
        person_bboxes {[type]} -- [description]

    Keyword Arguments:
        scores {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    keep_top_n = 3

    def sorted_key_func(lhs_data, rhs_data):
        lhs_bbox_area = utils.bbox.calculate_area(lhs_data[0])
        rhs_bbox_area = utils.bbox.calculate_area(rhs_data[0])
        return -1 if lhs_bbox_area >= rhs_bbox_area else -1

    indexes = range(len(person_bboxes))
    bboxes, indexes = zip(*sorted(zip(person_bboxes, indexes),
                                  key=functools.cmp_to_key(sorted_key_func)))
    bboxes = np.array(bboxes)
    if scores:
        scores_sorted = [scores[each_index] for each_index in indexes]
    return indexes[:keep_top_n], bboxes[:keep_top_n], scores_sorted[:keep_top_n]
```

**_References:_**

- [CSDN: python3 sorted 如何自定义排序标准](https://blog.csdn.net/u012436149/article/details/79952975)

- [stackoverflow: Python: sort a list and change another one consequently](https://stackoverflow.com/questions/2732994/python-sort-a-list-and-change-another-one-consequently)

<!--  -->
<br>

---

<br>
<!--  -->

# Tricks

## Finding the mode of a list

**_References:_**

- [stackoverflow: Finding the mode of a list](https://stackoverflow.com/a/10797913/4636081)

<!--  -->
<br>

---

<br>
<!--  -->

# re

[Python doc: re --- 正则表达式操作](https://docs.python.org/zh-cn/3/library/re.html)

- backslash: `\`
- Escape sequence: 转义序列

[Python Escape Sequence table](https://docs.python.org/2.0/ref/strings.html)

## raw string

> From Python's tutorial: When an '**r**' or '**R**' prefix is present, a character following a backslash is included in the string without change, and all backslashes are left in the string

> From stackoverflow: Any character following a backslash **is** part of raw string.

> @rivergold: 在编程语言中的`\`反斜杠会将其后的某些特殊字符进行转义，而`raw string`会是字符串中的`\`后面的字符得以保留。但是在 raw_string 中`\`不能单独出现，也就是说，其后必须得有字符，且不能是`'`和`"`

> @rivergold: 在编程语言中的字符串有转义序列，例如`\n`, `\t`等，在正则表达式中也有转义序列，例如`\d`, `\w`等等。在`re`中使用 raw string 的目的是为了在我们不想让字符串中的`\字符`被转义时少写`\`而使用的

E.g.

```python
x = '[]abc'
# 我们想找到是否含有[]
res = re.findall(r'\[\]', x)
print(res)
>>> ['[]']

res = re.findall(r'[]', x) # ERROR, 因为[]在正则表达式中有特殊的含义，表示字符集合
```

**_References:_**

- :thumbsup::thumbsup:[JournalDev: Python Raw String](https://www.journaldev.com/23598/python-raw-string)
- :triangular_flag_on_post::thumbsup::thumbsup::thumbsup:[stackoverflow: Why can't Python's raw string literals end with a single backslash?](https://stackoverflow.com/a/19654184/4636081)

---

## re.sub

### Replace part of string

E.g.

```python
in_text = '你好 再见'
partten = re.compile(r'([\w\u4e00-\u9fa5]{1})\s+([\u4e00-\u9fa5]{1})')
out_text = partten.sub(r'\1,\2', in_text)
print(out_text)
>>> 你好,再见
```

**_References:_**

- [stackoverflow: python regular expression “\1”](https://stackoverflow.com/questions/20802056/python-regular-expression-1/20802130)

<!--  -->
<br>

---

<br>
<!--  -->

# String Encode and Decode

- 字符集： 为每一个「字符」分配一个唯一的 ID，这个 ID 学名叫做：**码位** Code Point
- 编码规则：将码位转化为字节序列，将字节序列转化为码位的规则

E.g.

- Unicode: 字符集
- utf-8：编码规则

**_References:_**

- [知乎: Unicode 和 UTF-8 有什么区别？](https://www.zhihu.com/question/23374078/answer/24385963)

> @rivergold: encode: 将字符码位编码为字节序列；decode：将字节序列解码为字符码位。

> @rivergold: 人类定义了不同的字符集标准，例如：Unicode，ASCII， GB2312 等，每种字符集的编解码方式都不一样。当字符串所采用的字符集和编码方式不匹配，或者字节序列解码的方式不匹配时，都可能会出错。

Python 采用 Unicode 作为字符集标准，默认采用 utf-8 作为编码规则

E.g.

```python
# ~~~~~~
# Encode
# ~~~~~~
x = 'Hello' # Python字符串采用Unicode字符集
print(x.encode('utf-8')) # 采用utf-8规则编码
>>> b'Hello'
print(x.encode('ascii')) # 采用ascii规则编码
>>> b'Hello'

x = '你好' # Python字符串采用Unicode字符集
print(x.encode('utf-8')) # 采用utf-8规则编码
>>> b'\xe4\xbd\xa0\xe5\xa5\xbd'
print(x.encode('ascii')) # 采用ascii规则编码，ascii不支持中文
>>> UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)

# ~~~~~~
# Decode
# ~~~~~~
x = b'\xe4\xbd\xa0\xe5\xa5\xbd'
print(x.decode('utf-8')) # 采用utf-8规则解码
>>> 你好
print(x.decode('ascii')) # 采用ascii规则解码
>>> UnicodeDecodeError: 'ascii' codec can't decode byte 0xe4 in position 0: ordinal not in range(128)
```

> :triangular_flag_on_post::triangular_flag_on_post::triangular_flag_on_post:@rivergold: 在进行文件读写时，一定要指定`encoding=xxx`指定文件的编码方式。

## chardet

[Github](https://github.com/chardet/chardet)

A package for character encoding.

<!--  -->
<br>

---

<br>
<!--  -->

# Position argument & Keyword argument

## :triangular_flag_on_post:You can pass keyword argument into function with position argument

E.g.

```python
def fun(a, b):
    print(a + b)

fun(a=1, b=2)
>>> 3
```

**_References:_**

- [Blog: Keyword (Named) Arguments in Python: How to Use Them](https://treyhunner.com/2018/04/keyword-arguments-in-python/)

<!--  -->
<br>

---

<br>
<!--  -->

# When need to use staticmethod

> @rivergold: 当你需要编写的 standalone 函数需要放到类里面时（因为该函数和类有比较强的设计逻辑关系），你需要将这个 standalone 函数移到类里面并标明为`@staticmethod`

**_References:_**

- [stackoverflow: Why do we use @staticmethod?](https://stackoverflow.com/a/23508293/4636081)