***References:***

- [简书: 开发者应知道的编译原理和语言基础知识](https://www.jianshu.com/p/0913993a4c3f)
- [知乎: 学编译原理有什么好书？](https://www.zhihu.com/question/25868417)

# Basics

## ld (linker) search dynamic library path order

1. `RPATH` in the binary file
2. `LD_LIBRARY_PATH`
3. `/etc/ld.so.conf.d`
4. `/lib` or `/usr/lib`

***References:***

- [stackoverflow: Order in which library directories are searched and linked](https://stackoverflow.com/questions/36015224/order-in-which-library-directories-are-searched-and-linked)

- [stackoverflow: What is the order that Linux's dynamic linker searches paths in?](https://unix.stackexchange.com/a/367682)

- [Blog: linux动态库加载RPATH,RUNPATH 链接动态库](https://gotowqj.iteye.com/blog/1926771)
    *该博客所说的链接顺序是错的，但是其中其他的知识点可以参考*