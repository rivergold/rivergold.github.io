# Materials

- [知乎: LLVM 怎样入门和上手？](https://www.zhihu.com/question/20236606)

# Question

## Visitor

**_References:_**

- [LLVM Tutorial: My First Language Frontend with LLVM Tutorial](http://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)

---

## `::=`是什么含义

**_References:_**

- [stackoverflow: What does a double colon followed by an equals sign (::=) mean in programming documentation?](https://stackoverflow.com/questions/9196066/what-does-a-double-colon-followed-by-an-equals-sign-mean-in-programming-do)

<!--  -->
<br>

---

<br>
<!--  -->

# ENV

## Embeding llvm in CMake

```shell
cmake_minimum_required(VERSION 3.4.3)
project(SimpleProject)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Now build our tools
add_executable(simple-tool tool.cpp)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Link against LLVM libraries
target_link_libraries(simple-tool ${llvm_libs})
```

**_References:_**

- [LLVM doc: Embedding LLVM in your project](https://llvm.org/docs/CMake.html#embedding-llvm-in-your-project)
