# Concept

> 基于Halide思想，将算法的算法的数学表达与计算顺序（这里的计算顺序我的理解是计算的实现）分离。计算图表示了算法的数学表达；使用schedule实现算法的计算顺序。

## Schedule

> Using existing programming tools, writing high-performance image processing code requires sacrificing readability, portability, and
modularity. We argue that this is a consequence of conflating what
computations define the algorithm, with decisions about storage
and the order of computation. We refer to these latter two concerns
as the schedule, including choices of tiling, fusion, recomputation
vs. storage, vectorization, and parallelism.

## NNVM

***References:***

- [量子位: 陈天奇团队发布NNVM编译器，性能优于MXNet，李沐撰文介绍](https://zhuanlan.zhihu.com/p/29914989)

## Others

### NEON

[Github: NervanaSystems/neon](https://github.com/NervanaSystems/neon)

neon is Intel's reference deep learning framework committed to best performance on all hardware. Designed for ease-of-use and extensibility.

<!--  -->
<br>

***

<br>
<!--  -->

# Install

## Install Android TVM RPC

### TVM4J - Java Frontend for TVM Runtime

Required

- JDK
- Maven

Check JDK installed or not

```bash
java -version # this will check your jre version
javac -version # this will check your java compiler version(JDK) if you installed
```

- JRE: Java Runtime Environment
- JDK: Java Development Kit
- Maven: 一个项目管理和构建工具，主要做编译、测试、报告、打包、部署等操作完成项目的构建。

Ref [stackoverflow: How to tell if JRE or JDK is installed](https://stackoverflow.com/questions/22539779/how-to-tell-if-jre-or-jdk-is-installed)

***References:***

- [知乎: JRE 和 JDK 的区别是什么？](https://www.zhihu.com/question/20317448)

- [知乎: maven是干嘛的？](https://www.zhihu.com/question/20104186/answer/73797359)

- [并发编程网: Maven入门指南（一）](http://ifeve.com/maven-1/)

- [简书: java中的maven是干什么的](https://www.jianshu.com/p/3ed036e1c816)

Install maven

```bash
sudo apt install maven
mvn --version
```

Ref [Linuxize: How to Install Apache Maven on Ubuntu 18.04](https://linuxize.com/post/how-to-install-apache-maven-on-ubuntu-18-04/)

```bash
cd <tvm folder>
make jvmpkg
bash tests/scripts/task_java_unittest.sh
make jvminstall
```

When I run `bash ` occur error like followings,

```bash
Failed to execute goal org.apache.maven.plugins:maven-checkstyle-plugin:2.17:check (default) on project tvm4j-core: Failed during checkstyle configuration: cannot initialize module TreeWalker - Property 'cacheFile' does not exist, please check the documentation
```