# Introduction

## wrap C++ into Python == python bind to C++

means Python code can call C++ function

## Cython vs pybind11

**_Ref:_** [Stefans Welt: Cython, pybind11, cffi – which tool should you choose?](http://blog.behnel.de/posts/cython-pybind11-cffi-which-tool-to-choose.html)

## Wrap Python into C++

It is common to wrap C++ into Python, wrap Python into C++ is not uncommonly used.

**_Ref:_** [Python Doc: Embedding Python in Another Application](https://docs.python.org/3.7/extending/embedding.html)

<!--  -->
<br>

---

<br>
<!--  -->

# Setuptools

```shell
- package_name
    - __init__.py
    - sub_package_1
        - __init__.py
        - module_1a.py
        - module_1b.py
    - sub_package_2
        - __init__.py
        - module_2a.py
        - module_2b.py
    - data
        - data.txt
    reamdme.txt
```

```python
from setuptools import setup, find_packages
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    name='package_name',
    version="0.1",
    packages=find_packages(),
    # Uncomment to add install requires
    # install_requires=['opencv-python<=3.4.5', 'numpy'],
    ext_modules=cythonize(
        [
            Extension("package_name.*", ["package_name/**/*.py"]),
        ],
        build_dir="build",
        compiler_directives=dict(always_allow_keywords=True)),
    package_data={
        # If any package contains *.txt files, include them:
        '': ['*.txt'],
        # And include any *.dat files found in the 'data' subdirectory
        # of the 'package_name' package, also:
        'package_name': ['data/*.txt']
    },
    # Uncomment to add new command function to do something.
    # cmdclass=dict(build_ext=build_ext),
    author="rivergold",
    author_email="rivergold@126.com",
    zip_safe=False,
)
```

[Here](https://github.com/rivergold/Python-Packaging-Distribution-Example/tree/dev/build_python_to_so) is a example code to show how to use setuptools.

**_Ref:_** [setuptools DOC: Including Data Files](https://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files)

- [ ] `find_packages()` will include raw script .py and .so into distribution, how to exclude .py files? When comment `find_packages`, the distribution `bdist` will only contain `.so` but without data files.
      Maybe work: First build Python script into `.so`, then use built `.so` file to set a new package, and then write a new setup.py and make a distribution.
      Another way is to rewrite `build_py`. Ref to [stackoverflow: Exclude single source file from python bdist_egg or bdist_wheel](https://stackoverflow.com/a/50517893/4636081)

- [ ] `package_dir` how to use?

---

## `find_packages(<path>)`

Find all packages and add them into distribution.

**_Ref:_** [setuptools doc: Using find_packages()](https://setuptools.readthedocs.io/en/latest/setuptools.html#using-find-packages)

---

## `package_data`

Include data files. `xx: []` means that `xx` package should include what data files.

**注:** 例如你想把`package_name`下的`data/data.txt`文件包含在 distribution 中， `package_data`的值为`'package_name': ['data/*.txt']`， 而不是`'package_name': ['package_name/data/*.txt']`

---

## `Extension`

- [Python Doc: Extension arguments](https://docs.python.org/3/distutils/apiref.html?highlight=extension)

---

## Use `setup.py`

Run followings to get help

```shell
python setup.py --help-commands
```

```shell
Standard commands:
  build             build everything needed to install
  build_py          "build" pure Python modules (copy to build directory)
  build_ext         build C/C++ extensions (compile/link to build directory)
  build_clib        build C/C++ libraries used by Python extensions
  build_scripts     "build" scripts (copy and fixup #! line)
  clean             clean up temporary files from 'build' command
  install           install everything from build directory
  install_lib       install all Python modules (extensions and pure Python)
  install_headers   install C/C++ header files
  install_scripts   install scripts (Python or otherwise)
  install_data      install data files
  sdist             create a source distribution (tarball, zip file, etc.)
  register          register the distribution with the Python package index
  bdist             create a built (binary) distribution
  bdist_dumb        create a "dumb" built distribution
  bdist_rpm         create an RPM distribution
  bdist_wininst     create an executable installer for MS Windows
  upload            upload binary package to PyPI

Extra commands:
  rotate            delete older distributions, keeping N newest files
  develop           install package in 'development mode'
  setopt            set an option in setup.cfg or another config file
  saveopts          save supplied options to setup.cfg or other config file
  egg_info          create a distribution's .egg-info directory
  upload_sphinx     Upload Sphinx documentation to PyPI
  install_egg_info  Install an .egg-info directory for the package
  alias             define a shortcut to invoke one or more commands
  easy_install      Find/get/install Python packages
  bdist_egg         create an "egg" distribution
  test              run unit tests after in-place build
  build_sphinx      Build Sphinx documentation

usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: setup.py --help [cmd1 cmd2 ...]
   or: setup.py --help-commands
   or: setup.py cmd --help
```

**_Ref:_** [Getting Started With setuptools and setup.py: Using setup.py](https://pythonhosted.org/an_example_pypi_project/setuptools.html#using-setup-py)

<!-- ## `Setup`

- [Python Doc: Writing the Setup Script](https://docs.python.org/3/distutils/setupscript.html)

**_References:_**

- [Python doc: distutils.core.setup](https://docs.python.org/3/distutils/apiref.html?highlight=setup#distutils.core.setup) -->

---

## Distribute

**_Ref:_** [koala bear: Python application 的打包和发布——(上)](http://wsfdl.com/python/2015/09/06/Python%E5%BA%94%E7%94%A8%E7%9A%84%E6%89%93%E5%8C%85%E5%92%8C%E5%8F%91%E5%B8%83%E4%B8%8A.html)

```python
# Binary
python setup bdist
# Binary wheel
python setup bdist_wheel
# Source
python setup sdist
```

**_Ref:_** [stackoverflow: How to include package data with setuptools/distribute?](https://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute/14159430)

---

## Set package version

Edit your package `__init__.py`, and add `__version__ = vx.x`

**_References:_**

- [stackoverflow: Standard way to embed version into python package?](https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package)

<!--  -->
<br>

---

<br>
<!--  -->

# Cython

## Why and When to use

If you are Python background, and want to gain the ability to do efficient native C/C++ operations.

`Python` :arrow_forward: `C/C++`

## `.pyx`

- The `cython` command takes a `.py` or `.pyx` file and compiles it into a C/C++ file.

- The `cythonize` command takes a `.py` or `.pyx` file and compiles it into a C/C++ file. It then compiles the C/C++ file into an extension module which is directly importable from Python.

**_Ref:_** [Cython doc: Source Files and Compilation](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiling-from-the-command-line)

- `.pyx` can import C++ file

---

## Compiler will do optimization when compile `.py` into `.so`

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

<!--  -->
<br>

---

<br>
<!--  -->

# Compile `.py` into `.so`

**_Ref:_** [Jan Buchar Blog: Using Cython to protect a Python codebase](https://bucharjan.cz/blog/using-cython-to-protect-a-python-codebase.html)

## Example

1. Write your python scripts as **package**.

   ```bash
   - package_name
       - __init__.py
       - sub_package_1
           - __init__.py
           - module_1a.py
           - module_1b.py
       - sub_package_2
           - __init__.py
           - module_2a.py
           - module_2b.py
   ```

2. Write `setup.py`

   ```python
   from setuptools import setup, find_packages
   from setuptools.extension import Extension
   from Cython.Build import cythonize
   from Cython.Distutils import build_ext

   setup(
       name='your_package_name',
       version="0.1",
       # install_requires=['opencv-python<=3.4.5', 'numpy'],
       ext_modules=cythonize(
           [
               Extension("package_name.*", ["package_name/*.py"]),
               Extension("package_name.sub_package_1.*", ["package_name/sub_package_1/*.py"]),
               Extension("package_name.sub_package_2.*", ["package_name/sub_package_2/*.py"]),
               # 以上三行代码也可以整合为一行：
               # Extension("package_name.*", ["package_name/**/*.py"]),
           ],
           build_dir="build",
           compiler_directives=dict(
               always_allow_keywords=True),
           compiler_directives={'language_level' : "3"}),

       # -----Copy other data into build
       package_data={
           '<package name>.<where to put data>': ['<>/<>/*']
       },
       # -----

       author="your name",
       author_email="you email")
   ```

3. Build

   ```bash
   python setup.py build
   # Force build
   # python setup.py build --force
   ```

**_References:_**

- [Github Gist rivergold/setup_build_so.py](https://gist.github.com/rivergold/31195407e4dc95067bfd661797909dec)

## :triangular_flag_on_post::triangular_flag_on_post::triangular_flag_on_post:Specify Python 3 source in Cython's setup.py

When you use Python3, it is very important to set `compiler_directives={'language_level' : "3"}` into `cythonize` in `setup.py`.

Because division operation in Python2 cython will not convert int into float. So if you do this in your Pyhton script:

```python
def func():
    a = len('a')
    b = len('abc')
    return a / b
```

After compile it into dynamic lib `.so`, the result will be zero !!!

**_References:_**

- [stackoverflow: Cython optimize Python script int divided by int into zero [duplicate]](https://stackoverflow.com/questions/58832967/cython-optimize-python-script-int-divided-by-int-into-zero)

<!--  -->
<br>

---

<br>
<!--  -->

# Compile C++ into `.so` for Python

**_Ref:_** [Cython: Using C++ in Cython](https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html)

**_References:_**

- [Medium: Making your C library callable from Python by wrapping it with Cython](https://medium.com/@shamir.stav_83310/making-your-c-library-callable-from-python-by-wrapping-it-with-cython-b09db35012a3)

- [博客园: 非极大值抑制（NMS）的几种实现](https://www.cnblogs.com/king-lps/p/9031568.html)

- [segmentfault: Python 中使用 C 代码：以 Numpy 为例](https://segmentfault.com/a/1190000000479951)

<!--  -->
<br>

---

<!--  -->

## Disribute package with shared library

**_References:_**

- [stackoverflow: Distribute a Python package with a compiled dynamic shared library](https://stackoverflow.com/questions/37316177/distribute-a-python-package-with-a-compiled-dynamic-shared-library)

- [Github pytorch/pytorch: setup.py](https://github.com/pytorch/pytorch/blob/master/setup.py)

<!--  -->
<br>

---

<br>
<!--  -->

# pybind11

## Why and when to use

If you are C++ background, it is easy to use pybind11 to wrap C++ (make Python bind to C++).

`C++` :arrow_forward: `Python`

**PyTorch use pybind11 to create Python binding for C++ code.**

**_References:_**

- [PyTorch Doc: Writing the C++ Op](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-the-c-op)

- [Stefans Welt: Cython, pybind11, cffi – which tool should you choose?](http://blog.behnel.de/posts/cython-pybind11-cffi-which-tool-to-choose.html)

<!--  -->
<br>

---

<br>
<!--  -->

# Uninstall `python setup.py install`

1. `pip freeze | grep <package_name>`
2. `pip uninstall <package_name>`

**_References:_**

- :thumbsup:[stackoverflow: python setup.py uninstall](https://stackoverflow.com/a/12797865/4636081)

<!--  -->
<br>

---

<br>
<!--  -->

# Pypi

## Publish package into pypi

1. Write your packe code and test it
2. Write `setup.py`
3. `python setup.py bdist_wheel`
4. `twine upload dist/*`
5. Install `pip install <your package_name>`

**Note:** Pypi upload url is `https://upload.pypi.org/legacy/` and `https://test.pypi.org/legacy/`, you can run `twine upload --repository-url https://upload.pypi.org/legacy/ dist/*`

> @rivergold: Better use upload package into `https://test.pypi.org/legacy/` to test if the package is ok. Then release it into `https://upload.pypi.org/legacy/`

**_References:_**

- [Medium: How to upload your python package to PyPi](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)
- [Pypi.org FAQ: Why am I getting a "Filename or contents already exists" or "Filename has been previously used" error?](https://pypi.org/help/#file-name-reuse)
- [stackoverflow: not able to update my package on pypi.org](https://stackoverflow.com/q/50436171/4636081)

---

## Wheel name means what ?

```shell
<package_name>-<package_version>-<python_version>-<abi_version>-<platform>.whl
```

You can run `python setup.py bdist_wheel --help` get how to set these.

**_References:_**

- [stackoverflow: Wheel files : What is the meaning of “none-any” in protobuf-3.4.0-py2.py3-none-any.whl](https://stackoverflow.com/questions/46915070/wheel-files-what-is-the-meaning-of-none-any-in-protobuf-3-4-0-py2-py3-none-a?noredirect=1&lq=1)
