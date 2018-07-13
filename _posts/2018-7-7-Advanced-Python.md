Python is a language which is flexible and easy to learn. And it has lots of convenient modules and  packages.

In this article, I summarized some useful modules and packages I have used for mackine learning and data science. Hope it will give you some help or guidance:smiley:.

<!--  -->

<br>

***
<!--  -->

# Valuable Websites

- [python3-cookbook](http://python3-cookbook.readthedocs.io/zh_CN/latest/index.html)
- [Python Tips](http://book.pythontips.com/en/latest/index.html)
- [Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

<!--  -->

<br>

***
<!--  -->

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

**Note:** Please read **Fluent Python** book.

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

- `list.insert(index, element)`: Insert element into list
    ```python
    a = [1,2,3]
    a.insert(2, 0)
    print(a)
    >>> [1,2,0,3]
    ```

### `zip`

- Combine two list:
    ```python
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = list(zip(a, b))
    print(c)
    >>> [(1, 4), (2, 5), (3, 6)]
    ```
    **理解:** `zip`就是将多个list对应位置上的element分别组合起来
- Unzip a list of list/tuple
    ```python
    a = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    a1, a2, a3 = zip(*a)
    print(a1, a3, a3)
    >>> (1, 4, 7) (2, 5, 8), (3, 6, 9)
    ```

    **Note:** If you regard `a` as a matrix, `zip(*a)` will get each column of `a`.

    ***References:***
    - [stackoverflow: How to unzip a list of tuples into individual lists? [duplicate]](https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists/12974504)
    - [stackoverflow: What does the Star operator mean? [duplicate]](https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean)

## dict

- `dict1.update(dict2)`: Adds dictionary `dict2's` key-values pairs in to `dict1`. This function does not return anything.

- `dict.pop(<key>)`: Delate a key of a dict

## string

- `"` in str

    ```python
    a = '\"abc\"'
    print(a)
    >>> "abc"
    ```

- `str.replace`

    ```python
    <str>.replace(<old_str>, <new_str>)
    ```

    You can use it to delete character by replace `<old_str>` as `''`

    ***References:***

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

    ***References:***
    - [Python3.6 doc str.translate](https://docs.python.org/3/library/stdtypes.html?highlight=maketrans#str.translate)

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

    ***References:***s
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

    ***References:***
    - [stackoverflow: How to move a file in Python](https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python)

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

    ***References:***
    - [stackoverflow: Manually raising (throwing) an exception in Python](https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python)
    - [Python tutorial: 错误与异常](http://www.pythondoc.com/pythontutorial3/errors.html)
    - [RUNOOB.COM: Python异常处理](http://www.runoob.com/python/python-exceptions.html)

<br>

* * *

<br>

# Modules

## pickle

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


## collections

This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.

- Reference
    - [不可不知的Python模块: collections](http://www.zlovezl.cn/articles/collections-in-python/)

### `defaultdict`

***Referneces:***

- [简书：Python中collections.defaultdict()使用](https://www.jianshu.com/p/26df28b3bfc8)

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

***Reference:***

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

```python
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

## multiprocessing

***References:***

- [Blog: Python多进程库multiprocessing中进程池Pool类的使用](https://blog.csdn.net/jinping_shi/article/details/52433867)
- [Blog: Python 多进程 multiprocessing.Pool类详解](https://blog.csdn.net/SeeTheWorld518/article/details/49639651)

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

***References:***

- [stackoverflow: Python multiprocessing pool.map for multiple arguments](https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments)
- [Rawidn's Blog: Python Pool.map多参数传递方法](https://www.rawidn.com/posts/Python-multiprocessing-for-multiple-arguments.html)
- [CODE Q&A: Python多处理pool.map多个参数](https://code.i-harness.com/zh-CN/q/530d5e)

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

level:
    - `DEBUG`
    - `INFO`
    - `WARNING`
    - `ERROR`
    - `CRITICAL`

`CRITICAL` is the highest level, when you set level as `CRITICAL`, other level information will not be print. The default level is `WARNING`.

`format` will set the format of the logging information format. The default format is `%(levelname)s:%(name)s:%(message)s`. Other format seting can be get from [Python doc-LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes) and [简书: python3 logging 学习笔记](https://www.jianshu.com/p/4993b49b6888)

***References:***

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

***References:***

- [stackoverflow: How to find if directory exists in Python](https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python)
- [stackoverflow: How to delete a file or folder](https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder)


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
    ```python
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

### Copy array

```python
a = np.zeros((2,2))
b = np.copy(a)
```

**Note!** Numpy arrays work differently than lists do. When use `b = a[::]` in numpy, `b` is not a deep copy of a, but a view of a.

***References:***

- [stackoverflow: Numpy array assignment with copy](https://stackoverflow.com/questions/19676538/numpy-array-assignment-with-copy)

### Copy in numpy

***References:***

- [CSDN:【Python】numpy中的copy问题详解](https://blog.csdn.net/u010099080/article/details/59111207)

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

<br>

## OpenCV

Using `import cv2` to import OpenCV

### Install `opencv-python`

```shell
pip install opencv-python
```

***References:***

- [Pypi: opencv-python 3.4.1.15](https://pypi.org/project/opencv-python/)

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
    ***References:***
    - [PHYSIOPHILE: WHY OPENCV USES BGR (NOT RGB)](https://physiophile.wordpress.com/2017/01/12/why-opencv-uses-bgr-not-rgb/)

- When using OpenCV in IPython or Jupyter notebook, `cv2.imshow` and `cv2.waitKey(0)` cause crash

    Solution: Add `cv2.destroyAllWindows()`
    ```python
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
    ***References:***
    - [Github Opencv/opencv Issues: opencv cv2.waitKey() do not work well with python idle or ipython](https://github.com/opencv/opencv/issues/6452)


## Cython

**What is Cython?** The Cython language is a superset of the Python language that additionally supports calling C functions and declaring C types on variables and class attributes. This allows the compiler to generate very efficient C code from Cython code. 
 Cython是包含C数据类型的Python，Cython编译器会转化Python代码为C代码，这些C代码均可以调用Python/C的API

***References:***

- [Github: cython/cython](https://github.com/cython/cython)
- [Cython 0.28.2 documentation](http://docs.cython.org/en/latest/src/tutorial/)
- [Gitbook: Cython官方文档中文版](https://moonlet.gitbooks.io/cython-document-zh_cn/content/ch1-basic_tutorial.html)

## easydict

**EasyDict** allows to access dict values as attributes(works recursively).

```python
from easydict import EasyDict as edict
d = edict({'foo':3, 'bar':{'x':1, 'y':2}})
print(d.foo)
print(d.bar.x)
```

***Referneces:***

- [Github: makinacorpus/easydict](https://github.com/makinacorpus/easydict)
- [Github: endernewton/tf-faster-rcnn/lib/model/config.py](https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/model/config.py)

<br>

* * *

<br> 

# Python Tips

## Change python packages download source.

- **pip**
    - On Windows:
        - New a folder called `pip` under path `c:/user/<your user name>/`, and new a file name `pip.ini`, write followings in it:
            ```python
            [global]
            index-url = https://pypi.tuna.tsinghua.edu.cn/simple
            ```

    - On Linux:
        - New a folder called `.pip` under `~/`, create a file named `pip.conf` and write the same as Windows in the file.<br>

- **conda**
    Input followings in terminal

    ```python
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --set show_channel_urls yes
    ```

***Reference:***

- [csdn: 更改pip源至国内镜像，显著提升下载速度](http://blog.csdn.net/zhangchilei/article/details/53893002)

## Organize your python module

Assume that your module is like,

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

If `b.py` want to import method or variable in `a.py`,

```python
# b.py
from ..a import <method>
```

***Referneces:***

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

***References:***

- [python3-cookbook: 3.3 数字的格式化输出](http://python3-cookbook.readthedocs.io/zh_CN/latest/c03/p03_format_numbers_for_output.html)

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

***References:***

- [Python Tips: 13. Enumerate](http://book.pythontips.com/en/latest/enumerate.html)

## Create a directory/folder if it does not exist

```python
import os
if not os.path.exists(<directory_path>):
    os.makedirs(<directory_path>)
```

***Reference:*** 

- [stackoverflow: How can I create a directory if it does not exist?](https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist)

## How to run `pip` in python script?

```python
import pip
pip.main(['install', '-r'. 'requirements.txt'])
```

***Reference:***

- [stackoverflow: use “pip install/uninstall” inside a python script](https://stackoverflow.com/questions/12937533/use-pip-install-uninstall-inside-a-python-script)
- [stackoverflow: How to pip install packages according to requirements.txt from a local directory?](https://stackoverflow.com/questions/7225900/how-to-pip-install-packages-according-to-requirements-txt-from-a-local-directory)

## Parallel in Python

***Reference:***

- [知乎专栏： Python多核并行计算](https://zhuanlan.zhihu.com/p/24311810)

## C++ Embed Python

***Reference:***

- [Embedding Python in C/C++: Part I](https://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I)
- [CSDN Blog: C++调用python](http://blog.csdn.net/marising/article/details/2917892)

## Download image from url

***References:***

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

***References:***

- [segmentfault: Python 频繁请求问题: [Errno 104] Connection reset by peer](https://segmentfault.com/a/1190000007480913)


<br>

* * *

<br>

# Problems and Solutions:

- `UnicodeEncodeError: 'ascii' codec can't encode character '\u22f1' in position 242`
    - [解决Python3下打印utf-8字符串出现UnicodeEncodeError的问题](https://www.binss.me/blog/solve-problem-of-python3-raise-unicodeencodeerror-when-print-utf8-string/)

**Continuously updated...**