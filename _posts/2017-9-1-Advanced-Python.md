Python is a language which is flexible and easy to learn. And it has lots of convenient modules and  packages.

In this article, I summarized some useful modules and packages I have used for mackine learning and data science. Hope it will give you some help or guidance:smiley:.

<br>

# Valuable Websites
- [python3-cookbook](http://python3-cookbook.readthedocs.io/zh_CN/latest/index.html)
- [Python Tips](http://book.pythontips.com/en/latest/index.html)
- [Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

<br>

# Python Build-in
## Variables
Variable in Python is a label bound to a object, it means that variable is a reference of the object.<br>
If you are a C++ programmer, you can think variable in Python as a pointer in C++ which point to the object.<br>

**理解：** Python中的变量是对object的标注，也可以说是object的引用，而且这种引用是可以动态改变的。而C++中的Varible就是那个object, 而且Python中的reference与C++的reference是有不同的： C++的reference就代表了那个object，且不能改变，而Python中的reference(variable)可以改变其reference的object，是动态的。而且如果你是一个C++掌握者，Python的Variable更像是C++的pointer。
```python
a = [1, 2, 3]
b = a # b is a reference of object [1, 2, 3]
pritn(b)
>>> [1, 2, 3]
b = [1, 2] # now b is a reference of another object: [1, 2]
print(a)
>>> [1, 2, 3]
```

## Function parameters as references
Type of passing parameters to function:
- **call by value:** the function gets a copy of the argument
- **call by reference:** the function gets a pointer to the argument
- **call by sharing:** the function gets a copy of the reference of the argument, it means that the parameters inside the function become a **alias** of the arguments.

In Python, the function gets a copy of the arguments, but the arguments are always references.<br>
**理解：** Python的函数传参，函数获得的是实参object的引用的拷贝<br>

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
    ***Reference:***
    - [Python doc: 2. Built-in Functions: zip](https://docs.python.org/3/library/functions.html#zip)
## dict
- `dict1.update(dict2)`: Adds dictionary `dict2's` key-values pairs in to `dict1`. This function does not return anything.
- `dict.pop(<key>)`: Delate a key of a dict

## string
- Remove punctuation(标点符号) in a string using `str.translate(table)`
    ```python
    import string
    trans_table = str.maketrans('', '', str.punctuation)
    data = 'this is a test!!!'
    data.translate(trans_table)
    print(data)
    >>> this is a test
    ```
***References:***
- [Python3.6 doc str.translate](https://docs.python.org/3/library/stdtypes.html?highlight=maketrans#str.translate)

## Errors and Exceptions
### Raising exceptions
The `raise` statement allows the programmer to force a specified exception to occur.
```python
raise NameError('Error')
```

***References:***
- [stackoverflow: Manually raising (throwing) an exception in Python](https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python)
- [Python tutorial: 错误与异常](http://www.pythondoc.com/pythontutorial3/errors.html)

<br>

* * *

<br>

# Modules:
## pickle:
1. What is pickle?

    [Offical introduction](https://docs.python.org/3/library/pickle.html) about pickle is

    >The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.

    In my view, `pickel` means _a thick cold sauce that is made from pieces of vegetables preserved in vinegar_. The implication is that you can store things for a long time. Pickel module provide you a good way to store your python data into disk, which is a persistence management. Lost of dataset is save as this way.


2. Common function:
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
        ```
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


## collections
This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.

- Reference
    - [不可不知的Python模块: collections](http://www.zlovezl.cn/articles/collections-in-python/)


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

**Reference**
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
    随机数种子seed只有一次有效，在下一次调用产生随机数函数前没有设置seed，则还是产生随机数。

    ***References:***
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
```
from enum import Enum
class Color(Enum):
    BLUE = 0
    GREEN = 1
    RED = 2
# get a list of this enum
color = [x.value for x in Color]
```

***References:***
- [stackoverflow: How to get all values from python enum class?](https://stackoverflow.com/questions/29503339/how-to-get-all-values-from-python-enum-class)

<br>

* * *

<br>

# Packages:
## Matplotlib

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
    ```
    r = np.random.RandomState(123)
    r.randn(2,2)
    ```

***References:***
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

    ***References:***
    - [Stackoverflow: convert numpy array to 0 or 1](https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1)

### Create array
- `np.meshgrid(x1, x2, ..., xn)`: Return coordinate matrices from coordinate vectors.
    ```python
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    ```
    **Note:** This function is used often when you want to draw your prediction model's decision boundary.

    ***References:***
    - [deeplearning.ai: Planar data classification with one hidden layer/planar_utils.py](https://github.com/XingxingHuang/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week3/Planar%20data%20classification%20with%20one%20hidden%20layer/planar_utils.py#L7)

- Copy array
    ```
    a = np.zeros((2,2))
    b = np.copy(a)
    ```
    **Note!** Numpy arrays work differently than lists do. When use `b = a[::]` in numpy, `b` is not a deep copy of a, but a view of a.

    ***References:***
    - [stackoverflow: Numpy array assignment with copy](https://stackoverflow.com/questions/19676538/numpy-array-assignment-with-copy)

<br>

## OpenCV
Using `import cv2` to import OpenCV
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

***References***:
- [OpenCV doc: warpAffine](https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)
- [stackoverflow: Rotate an image without cropping in OpenCV in C++](https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c)

<br>

* * *

<br> 

# Python Tips:
## Change python packages download source.
pip
: - On Windows:
    - New a folder called `pip` under path `c:/user/<your user name>/`, and new a file name `pip.ini`, write followings in it:
        ```python
        [global]
        index-url = https://pypi.tuna.tsinghua.edu.cn/simple
        ```

: - On Linux:
    - New a folder called `.pip` under `~/`, create a file named `pip.conf` and write the same as Windows in the file.<br>

conda
: Input followings in terminal
: ```python
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --set show_channel_urls yes
  ```

***Reference:***
- [csdn: 更改pip源至国内镜像，显著提升下载速度](http://blog.csdn.net/zhangchilei/article/details/53893002)

## Organize your python module
Assume that your module is like,
```
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
If `b.py` want to import method or variable in `a.py`,
```
# b.py
from ..a import <method>
```

***Referneces:***
- [python3-cookbook: 10.3 使用相对路径名导入包中子模块](http://python3-cookbook.readthedocs.io/zh_CN/latest/c10/p03_import_submodules_by_relative_names.html)

## Python script import from parent directory
Assume you have some scripts like
```
src
  |- part_1
      |- a.py
  |- part_2
      |- b.py
```
And `b.py` want to use method or variable in `a.py`, one solution is to add parent path into `sys.path`
```
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

***References:***
- [python3-cookbook: 3.3 数字的格式化输出](http://python3-cookbook.readthedocs.io/zh_CN/latest/c03/p03_format_numbers_for_output.html)

## `enumerate` using details
Common use is like followings,<br>
```python
for counter, value in enumerate(some_list):
    print(counter, value)
```

`enumerate(iterable, start=0)`, optional parameters `start` decide the start number of counter,<br>
```python
a = ['apple', 'banana', 'orange']
for counter, value in enumerate(a, 1)
    print(counter, value)
>>> 1 apple
>>> 2 banana
>>> 3 orange
```

***References:***
- [Python Tips: 13. Enumerate](http://book.pythontips.com/en/latest/enumerate.html)


## Create a directory/folder if it does not exist
```python
import os
if not os.path.exists(<directory_path>):
    os.makedirs(<directory_path>)
```

Reference: 
- [stackoverflow: How can I create a directory if it does not exist?](https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist)


## How to run `pip` in python script?
```python
import pip
pip.main(['install', '-r'. 'requirements.txt'])
```

Reference
: - [stackoverflow: use “pip install/uninstall” inside a python script](https://stackoverflow.com/questions/12937533/use-pip-install-uninstall-inside-a-python-script)
: - [stackoverflow: How to pip install packages according to requirements.txt from a local directory?](https://stackoverflow.com/questions/7225900/how-to-pip-install-packages-according-to-requirements-txt-from-a-local-directory)


## Parallel in Python
Reference
: - [知乎专栏： Python多核并行计算](https://zhuanlan.zhihu.com/p/24311810)


## C++ Embed Python
### Reference
- [Embedding Python in C/C++: Part I](https://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I)

- [CSDN Blog: C++调用python](http://blog.csdn.net/marising/article/details/2917892)

<br>

* * *

<br>

# Problems and Solutions:
- `UnicodeEncodeError: 'ascii' codec can't encode character '\u22f1' in position 242`
    - [解决Python3下打印utf-8字符串出现UnicodeEncodeError的问题](https://www.binss.me/blog/solve-problem-of-python3-raise-unicodeencodeerror-when-print-utf8-string/)

Continuously updated...
