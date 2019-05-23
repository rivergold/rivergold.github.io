***References:***

- [简书: 开发者应知道的编译原理和语言基础知识](https://www.jianshu.com/p/0913993a4c3f)
- [知乎: 学编译原理有什么好书？](https://www.zhihu.com/question/25868417)

# Basics

## Compile Process

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-7626b4e5017b41ac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="50%">
</p>

### Code generation

**Ref:** [Wiki: Code generation (compiler)](https://en.wikipedia.org/wiki/Code_generation_(compiler))

<!--  -->
<br>

***
<!--  -->

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

<!--  -->
<br>

***
<!--  -->

## 并发与并行 (Concurrent & Parallel)

> Rob Pike大神关于两者的阐述：“并发关乎结构，并行关乎执行”

并发: 多个任务可以同时存在
并行: 多个任务可以同时执行

代码可以写成并发的，但是如果cpu不支持并行，就无法并行

Ref [知乎: 并发与并行的区别？](https://www.zhihu.com/question/33515481)

<!--  -->
<br>

***

<br>
<!--  -->

# LLVM

The **LLVM** Compiler Infrastructure.

典型的编译器架构:

- Frontend(Parser): 负责将源码解析成语法书
- Optimizer: 负责寻找程序逻辑中有没有可以简化的来提升执行速度
- Backend(Code Generator): 生成低阶的机器码

LLVM IR: LLVM定义的一种通用程序中间表示方法

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-0ced79013b8632aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="80%">
</p>

Frontend 把原始语言的逻辑翻译成LLVM IR，Optimizer 把 LLVM IR 整理成效率更好的 LLVM IR、Backend 拿到 LLVM IR 來生成机器目标平台的机器语言。

LLVM的Optimizer是由多个**Pass**组成

一个LLVM Pass的input

Ref [Medium: 編譯器 LLVM 淺淺玩](https://medium.com/@zetavg/%E7%B7%A8%E8%AD%AF%E5%99%A8-llvm-%E6%B7%BA%E6%B7%BA%E7%8E%A9-42a58c7a7309)

<!--  -->
<br>

***

<br>
<!--  -->

# Tricks

## Cross compile to Windows from Liunx

Ref [Array Fire: Cross Compile to Windows From Linux](https://arrayfire.com/cross-compile-to-windows-from-linux/)

<!--  -->
<br>

***

<br>
<!--  -->