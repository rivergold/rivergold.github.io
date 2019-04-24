# Q&A

## If I have cuda but not have GPU, can I build OpenCV or other libs with CUDA ?

当库编译时使用Cuda时，是否需要GPU做计算？

暂时的理解：不需要，只需要把对应的Cuda库链接就可以了。当真正执行时，在调用GPU。

## Can Python `.pyc` run ?

```python
python <yourfile.pyc>
```

And it can be decompile into `.py`

***References:***

- [StackExchange: How to run a .pyc (compiled python) file?](https://askubuntu.com/questions/153823/how-to-run-a-pyc-compiled-python-file)

## Ubuntu 16.04 not have`libpython3.6-dev`

Maybe can download from [here](https://stackoverflow.com/questions/43621584/why-cant-i-install-python3-6-dev-on-ubuntu16-04). But not have a try.

## 库和框架的区别

- [Blog: 库和框架的区别](https://www.cnblogs.com/xuld/archive/2011/02/20/1958933.html)
- [知乎： 库，框架，架构，平台，有什么明确的区别？](https://www.zhihu.com/question/29643471)

## Python `python -m <package>` and `if __name__ == '__main__'`

***References:***

- [stackoverflow: What does it mean to “run library module as a script” with the “-m” option? [duplicate]](https://stackoverflow.com/questions/46319694/what-does-it-mean-to-run-library-module-as-a-script-with-the-m-option)
- [stackoverflow: What does if __name__ == “__main__”: do?](https://stackoverflow.com/questions/419163/what-does-if-name-main-do)

## Google Coding style

Base on [Google coding style](https://google.github.io/styleguide/)

### C++

Private member name: `name_`

```c++
class TableInfo {
  ...
 private:
  string table_name_;  // OK - underscore at end.
  string tablename_;   // OK.
  static Pool<TableInfo>* pool_;  // OK.
};
```

Ref [Google C++ Style Guide: Variable Names](https://google.github.io/styleguide/cppguide.html#Variable_Names)

### Python

Private member name: `_name`