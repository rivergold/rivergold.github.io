Python is a language which is flexible and easy to learn. And it has lots of convenient modules and  packages.

In this article, I summarized some useful modules and packages I have used for mackine learning and data science. Hope it will give you some help or guidance:smiley:.

<br>

# Valuable Websites
- [python3-cookbook](http://python3-cookbook.readthedocs.io/zh_CN/latest/index.html)

- [Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/)

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

How to rename a folder?
: - Using `os.rename(<old folder path>, <new folder path>)`
  - Using `shutil.move(<old folder path>, <new folder path>)`

**Reference**
: - [stackoverflow: How to rename a file using Python](https://stackoverflow.com/questions/2491222/how-to-rename-a-file-using-python)
  - [stackoverflow: When using python os.rmdir, get PermissionError: [WinError 5] Access is denied](https://stackoverflow.com/questions/36360167/when-using-python-os-rmdir-get-permissionerror-winerror-5-access-is-denied)


## random
This module implements pseudo-random number generators for various distributions.

- `random.shuffle(x)`
    Shuffle the squence x in place<br>
    ```python
    a = [1, 2, 3, 4]
    random.shuffle(a)
    >>> [2, 1, 4, 3]
    ```

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
### Common Function
- Dot product: `np.dot(x1, x2)`
    Dot product of two arrays.
    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors (without complex conjugation). For N dimensions it is a sum product over the last axis of a and the second-to-last of b:
- Outer product: `np.outer(x1, x2)`
- Elementwise multiplication: `np.multiply(x1, x2)`

**Note:** `np.dot()` performs a matrix-matrix or matrix-vector multiplication.

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

Reference
: - [csdn: 更改pip源至国内镜像，显著提升下载速度](http://blog.csdn.net/zhangchilei/article/details/53893002)


## How print in one with dynamically refresh in Python?
- Python3
    ```python
    print(data, end'\r', flush=True)
    ```

- Python2
    ```python
    import sys
    sys.stdout.write('.')
    # or from Python 2.6 you can import the `print` function from Python3
    from __future__ import print_function
    ```


## `enumerate` using details [(\*ref)](http://book.pythontips.com/en/latest/enumerate.html)
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


## Create a directory/folder if it does not exist
```python
import os
if not os.path.exists(<directory_path>):
    os.makedirs(<directory_path>)
```

Reference
: - [stackoverflow: How can I create a directory if it does not exist?](https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist)


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

# Problems and Solutions:
- `UnicodeEncodeError: 'ascii' codec can't encode character '\u22f1' in position 242`
    - [解决Python3下打印utf-8字符串出现UnicodeEncodeError的问题](https://www.binss.me/blog/solve-problem-of-python3-raise-unicodeencodeerror-when-print-utf8-string/)

Continuously updated...
