# Setuptools

## `Extension`

- [Python Doc: Extension arguments](https://docs.python.org/3/distutils/apiref.html?highlight=extension)

## `Setup`

- [Python Doc: Writing the Setup Script](https://docs.python.org/3/distutils/setupscript.html)

***References:***

- [Python doc: distutils.core.setup](https://docs.python.org/3/distutils/apiref.html?highlight=setup#distutils.core.setup)

### `find_packages()`

***Ref:*** [setuptools doc: Using find_packages()](https://setuptools.readthedocs.io/en/latest/setuptools.html#using-find-packages)

<!--  -->
<br>

***

<br>
<!--  -->

# Cython

## `.pyx`

- The `cython` command taks a `.py` or `.pyx` file and compiles it into a C/C++ file.

- The `cythonize` command takes a `.py` or `.pyx` file and compiles it into a C/C++ file. It then compiles the C/C++ file into an extension module which is directly importable from Python.

***Ref:*** [Cython doc: Source Files and Compilation](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiling-from-the-command-line)

- `.pyx` can import C++ file

<!--  -->
<br>

***

<br>
<!--  -->

# Compile `.py` into `.so`

***Ref:*** [Jan Buchar Blog: Using Cython to protect a Python codebase](https://bucharjan.cz/blog/using-cython-to-protect-a-python-codebase.html)

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
                always_allow_keywords=True)),

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

***References:***

- [Github Gist rivergold/setup_build_so.py](https://gist.github.com/rivergold/31195407e4dc95067bfd661797909dec)

<!--  -->
<br>

***

<br>
<!--  -->

# Compile C++ into `.so` for Python

***Ref:*** [Cython: Using C++ in Cython](https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html)

***References:***

- [Medium: Making your C library callable from Python by wrapping it with Cython](https://medium.com/@shamir.stav_83310/making-your-c-library-callable-from-python-by-wrapping-it-with-cython-b09db35012a3)

- [博客园: 非极大值抑制（NMS）的几种实现](https://www.cnblogs.com/king-lps/p/9031568.html)

- [segmentfault: Python 中使用 C 代码：以 Numpy 为例](https://segmentfault.com/a/1190000000479951)

<!--  -->
<br>

***
<!--  -->

## Disribute package with shared library

***References:***

- [stackoverflow: Distribute a Python package with a compiled dynamic shared library](https://stackoverflow.com/questions/37316177/distribute-a-python-package-with-a-compiled-dynamic-shared-library)

- [Github pytorch/pytorch: setup.py](https://github.com/pytorch/pytorch/blob/master/setup.py)