# Setuptools

## `Extension`

- [Python Doc: Extension arguments](https://docs.python.org/3/distutils/apiref.html?highlight=extension)

## `Setup`

- [Python Doc: Writing the Setup Script](https://docs.python.org/3/distutils/setupscript.html)

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