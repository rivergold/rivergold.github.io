# Good References

## Book

### An Introduction to Modern CMake

- [GitBook](http://cliutils.gitlab.io/modern-cmake/)
- [中文版](https://xiazuomo.gitbook.io/modern-cmake-chinese/)

## Blog

- :thumbsup::thumbsup::thumbsup:[Medium: CMake Tutorial](https://medium.com/@onur.dundar1/cmake-tutorial-585dd180109b): An overall introduction for cmake.
- [Green Memory: CMake 学习笔记](https://blog.gmem.cc/cmake-study-note)

<!--  -->
<br>

---

<br>
<!--  -->

# Key Concept

## `.cmake`

`.cmake` file can be considered as a module for CMakeLists.txt. CMakeLists.txt uses `include` command to import `.cmake`.

In `.cmake`, you can write some cmake command, define function and MACRO.

**_References:_**

- [stackoverflow: What is the purpose of a .cmake file?](https://stackoverflow.com/questions/46456498/what-is-the-purpose-of-a-cmake-file)
- [CSDN: cmake 中的 include 指令（.cmake 文件/MACRO 宏/function 函数）](https://blog.csdn.net/qq_38410730/article/details/102677143)

<!--  -->
<br>

---

<br>
<!--  -->

# Variable

```shell
set(MY_VARIABLE "value")
```

The names of variables are usually all caps.

## Cache Variable

```shell
set(MY_CACHE_VARIABLE "VALUE" CACHE STRING "Description")
```

## Environment Variables

```shell
# set
set(ENV{variable_name} value)
# get
$ENV{variable_name}
```

## Common Variable

- **`PROJECT_SOURCE_DIR`**: Top level source directory for the current project.

  ```makefile
  project(<project name>)
  ```

- **`CMAKE_SOURCE_DIR`**: This is the full path to the top level of the current CMake source tree. For an in-source build, this would be the same as `CMAKE_BINARY_DIR`.

- **`CMAKE_CURRENT_SOURCE_DIR`**: The path to the source directory currently being processed.

- **`CMAKE_BINARY_DIR`**: The path to the top level of the build tree.

- **`get_filename_component(PARENT_DIR ${MYPROJECT_DIR} DIRECTORY)`**: Get parent path

  **_Ref:_** [stackoverflow: CMake : parent directory ?](https://stackoverflow.com/questions/7035734/cmake-parent-directory)

- **`PROJECT_BINARY_DIR`**: Full path to build directory for project.

  - in-source: Top level source directory for the current project.
  - out-of-source: The directory of `cmake` command run.

  **_Ref:_** [CSDN: CMake PROJECT_BINARY_DIR 和 PROJECT_SOURCE_DIR 区别](https://blog.csdn.net/sukhoi27smk/article/details/46388711)

<!-- - `CMAKE_BINARY_DIR`: The path to the top level of the build tree.
    ```bash
    mkdir build && cd build
    cmake ..
    # ${CMAKE_BINARY_DIR} is build/
    ``` -->

**_References:_**

- [维基教科书: CMake 入門/Out-of-source Build](https://zh.wikibooks.org/zh/CMake_%E5%85%A5%E9%96%80/Out-of-source_Build)

## &clubs; Defferent between `PROJECT_variable` and `CMAKE_variable`

`CMAKE_SOURCE_DIR` does indeed refer to the folder where the top-level CMakeLists.txt is defined.

However, `PROJECT_SOURCE_DIR` refers to the folder of the CMakeLists.txt containing the most recent project() command. When sub-project `CMakeLists.txt` has `prject(<project name>)`, `PROJECT_SOURCE_DIR` in sub-project will be changed into sub-project path.

> This difference applies to all PROJECT*<var> vs CMAKE*<var> variables.

**理解:** `PROJECT_`是会基于`prject()`的，而`CMAKE_`不会。

**_Ref:_** [stackoverflow: Are CMAKE_SOURCE_DIR and PROJECT_SOURCE_DIR the same in CMake?](https://stackoverflow.com/a/32030551/4636081)

## Properties

THe other way for CMake to store information.

```shell
set_property(TARGET TargetName
             PROPERTY CXX_STANDARD 11)
```

<!--  -->
<br>

---

<br>
<!--  -->

# Scope

## `add_subdirectory` scope

As mentioned in the documentation of the [set](https://cmake.org/cmake/help/latest/command/set.html) command, each directory added with `add_subdirectory` or each function declared with `function` creates a new scope.

The new child scope inherits all variable definitions from its parent scope. Variable assignments in the new child scope with the set command will only be visible in the child scope unless the `PARENT_SCOPE` option is used.

`add_subdirectory`的 scope 会继承父 scope。

**_References:_**

- [stackoverflow: cmake variable scope, add_subdirectory](https://stackoverflow.com/a/6891527/4636081)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:set

## Set cache variable

Cache variable has default value, and you can use `cmake -DVARIABLENAME=xxx` to change it.

```shell
set(<variable_name> <variable_value> CACHE <type> <docstring> FORCE)
# E.g.
set(My_VARIABLE "example" CACHE STRING "This is an example" FORCE)
```

- `FORCE`: If `FORCE` is specified, the value of the cache variable is set, even if the variable is already in the cache.

**_References:_**

- [CMake doc: set](https://cmake.org/cmake/help/latest/command/set.html)
- [An Introduction to Modern CMake: Cache Variables](https://cliutils.gitlab.io/modern-cmake/chapters/basics/variables.html)
- [stackoverflow: Why do I need FORCE to override a CMake option?](https://stackoverflow.com/questions/39588836/why-do-i-need-force-to-override-a-cmake-option)

<!--  -->
<br>

---

<br>
<!--  -->

# Common Command

## `CMAKE_PREFIX_PATH`

list of directories path to tell cmake where to search `find_package()`, `find_program()`, `find_library(), find_file()`

---

## `find_package()`

### :triangular_flag_on_post: Use pkg-config to find package

```makefile
set(CMAKE_PREFIX_PATH /usr/local/Cellar/glfw/3.2.1/lib/)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED GLFW3)
message("GLFW3 ${GLFW3_INCLUDE_DIRS}")
include_directories(${GLFW3_INCLUDE_DIRS})
# <XXX>_INCLUDE_DIRS
# <XXX>_LINK_LIBRARIES
```

Ref [stackoverflow: What is the proper way to use `pkg-config` from `cmake`?](https://stackoverflow.com/a/29316084)

**_References:_**

- [CMake doc: FindPkgConfig](https://cmake.org/cmake/help/v3.14/module/FindPkgConfig.html)

---

## `add_subdirectory`

Adds a subdirectory to the build.

### Problems & Solutions

#### When add not subdirectory into CMake Project, occur `add_subdirectory not given a binary directory but the given source, When specifying an out-of-tree source a binary directory must be explicitly specified.`

**Solution**

Set `binary directory`

```shell
add_subdirectory(<subdirectory_absolution_path> output.out)
```

**_Ref:_** [CSDN: cmake:用 add_subdirectory()添加外部项目文件夹](https://blog.csdn.net/10km/article/details/51889385)

---

## `add_definitions`

Adds -D define flags to the compilation of source files.

When the cpp file have `#ifdef` like followings:

```c++
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif
```

You can use `add_definitions` to control it.

```makefile
add_definitions(-DNDEBUG)
```

Ref [stackoverflow: CMake: How to pass preprocessor macros](https://stackoverflow.com/a/9639605)

---

## `file`

### List all `cpp` file as src_list to build

```shell
file(GLOB SRC_LIST "*.cpp")
```

**_References:_**

- [CMake Doc: file](https://cmake.org/cmake/help/v3.14/command/file.html)
- [stackoverflow: Automatically add all files in a folder to a target using CMake?](https://stackoverflow.com/a/3201211/4636081)

---

## `install`

**_Ref:_** [博客园: CMake 手册详情 install](https://www.cnblogs.com/coderfenghc/archive/2012/08/12/2627561.html)

**_References:_**

- [维基教科书: CMake 入門/輸出位置與安裝](https://zh.wikibooks.org/zh/CMake_%E5%85%A5%E9%96%80/%E8%BC%B8%E5%87%BA%E4%BD%8D%E7%BD%AE%E8%88%87%E5%AE%89%E8%A3%9D)

---

## `include`

- [CMake Doc `include`](https://cmake.org/cmake/help/v3.15/command/include.html?highlight=include)

Load and run CMake code from a file or module.

---

## `set`

---

## `include_directories`

All targets in this CMakeList, as well as those in all subdirectories added after the point of its call, will have the path `include_path` added to their include path.

## `target_include_directories`

has target scope—it adds `include_path` to the include path for `target`

**_Ref:_** [stackoverflow: What is the difference between include_directories and target_include_directories in CMake?](https://stackoverflow.com/a/31969632/4636081)

---

## `target_link_libraries`

When someone want to compile other target depend on this target, `PUBLIC`, `PRIVATE` and `INTERFACE` will have different influence.

E.g.

```shell
add_library(archive archive.cpp)
target_compile_definitions(archive INTERFACE USING_ARCHIVE_LIB)

add_library(serialization serialization.cpp)
target_compile_definitions(serialization INTERFACE USING_SERIALIZATION_LIB)

add_library(archiveExtras extras.cpp)
target_link_libraries(archiveExtras PUBLIC archive)
target_link_libraries(archiveExtras PRIVATE serialization)
# archiveExtras is compiled with -DUSING_ARCHIVE_LIB
# and -DUSING_SERIALIZATION_LIB

add_executable(consumer consumer.cpp)
# consumer is compiled with -DUSING_ARCHIVE_LIB
target_link_libraries(consumer archiveExtras)
```

Because `archive` is a `PUBLIC` dependency of `archiveExtras`, the usage requirements of it are propagated to `consumer` too. Because `serialization` is a PRIVATE dependency of `archiveExtras`, the usage requirements of it are not propagated to `consumer`.

**:triangular_flag_on_post:Rule of Thumb:**

When and which to use:

- `PUBLIC`: Your target source files and header files include the link library's header
- `PRIVATE`: Only your target source files but not header files include the link library's header
- `INTERFACE`: Only your target header files but not source files include the link library;s header

**_References:_**

- [stackoverflow: CMake target_link_libraries Interface Dependencies](https://stackoverflow.com/questions/26037954/cmake-target-link-libraries-interface-dependencies)

- [CMake Doc: cmake-buildsystem(7) - Transitive Usage Requirements](https://cmake.org/cmake/help/v3.15/manual/cmake-buildsystem.7.html#transitive-usage-requirements)

---

## `add_custom_command` and `add_custom_target`

When you want to run third-party command in CMake, you may need `add_custom_command` and `add_custom_target`.

For example, in PyTorch source `cmake/Codegen.cmake`, it use Python to run `aten/src/ATen/gen.py` to generate some `ATen` source `.cpp` and `.h` from template to achieve generic programming.

:thumbsup:[Here](https://gist.github.com/socantre/7ee63133a0a3a08f3990) is a common use example of `add_custom_command` and `add_custom_target`:

```shell
set(LIBFOO_TAR_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/include/foo/foo.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/foo/foo_utils.h"
)

add_custom_command(OUTPUT ${LIBFOO_TAR_HEADERS}
  COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_SOURCE_DIR}/libfoo/foo.tar"
  COMMAND ${CMAKE_COMMAND} -E touch ${LIBFOO_TAR_HEADERS}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/foo"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/libfoo/foo.tar"
  COMMENT "Unpacking foo.tar"
  VERBATIM
)

add_custom_target(libfoo_untar DEPENDS ${LIBFOO_TAR_HEADERS})

add_library(foo INTERFACE)
target_include_directories(foo INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include/foo")
target_link_libraries(foo INTERFACE ${FOO_LIBRARIES})
```

**核心理解:** 由于 cmake 的构建系统需要以 target 为基础，但是执行第三方命名时（例如，创建新的文件），该过程没有形成 target。因此为了将此 command 纳入到 cmake 的构建系统中，首先，采用`add_custom_command`执行该 command，并以 OUTPUT 作为输出的表达，**这仅仅是一个对输出的表达！**；然后，通过`add_custom_target`将之前的输出表达纳入到 cmake 的 target 构建体系中。

针对于这个问题， [Sam Thursfield's Blog: CMake: dependencies between targets and files and custom commands](https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/)写的很好。

**_References:_**

- :thumbsup:[GitHub Gist: socantre/CMakeLists.txt](https://gist.github.com/socantre/7ee63133a0a3a08f3990)

- :thumbsup:[Sam Thursfield's Blog: CMake: dependencies between targets and files and custom commands](https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/)

- [GitHub Gist: baiwfg2/CMakeLists.txt](https://gist.github.com/baiwfg2/39881ba703e9c74e95366ed422641609): references in it is very good.

---

## :triangular_flag_on_post:CMake Generator

A generator may be selected via the `cmake(1)` -G option, interactively in `cmake-gui(1)`, or via the `CMAKE_GENERATOR` environment variable.

**_References:_**

- :thumbsup:[CMake doc: CMAKE_GENERATOR](https://cmake.org/cmake/help/latest/variable/CMAKE_GENERATOR.html)

E.g., set cmake generator as ninja

```shell
# Method-1: cmake
cmake -GNinja ..
# Method-2: cmake environment
# Edit CMakeLists.txt
set(CMAKE_GENERATOR "Ninja")
```

<!--  -->
<br>

---

<br>
<!--  -->

# CMake Project Structure

```shell
- Project
    - .gitignore
```

**`cmake` folder**

`cmake` folder: has all of your helper modules. This is where your `Find*.cmake` files go.

To add this folder to your CMake path:

```shell
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
```

<!--  -->
<br>

---

<br>
<!--  -->

# Tips

## The theory of find package

**_Ref:_** [CSDN find_package 与 CMake 如何查找链接库详解](https://blog.csdn.net/bytxl/article/details/50637277)

<!--  -->
<br>

---

<!--  -->

## Set cmake to find Python

`find_package( PythonInterp 3.5 REQUIRED )`

<!--  -->
<br>

---

<!--  -->

## Set cmake to find boost

```makefile
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS system)
```

**_References:_**

- [stackoverflow: How do you add Boost libraries in CMakeLists.txt?](https://stackoverflow.com/a/6646518/4636081)

<!--  -->
<br>

---

<!--  -->

## Set cmake to find OpenCV

```makefile
find_package( OpenCV REQUIRED )
include_directories( $(OpenCV_INCLUDE_DIRS) )

target_link_libraries( <your target> ${OpenCV_LIBS})
```

**_References:_**

- [OpenCV doc: Using OpenCV with gcc and CMake](https://docs.opencv.org/3.0.0/db/df5/tutorial_linux_gcc_cmake.html)

<!--  -->
<br>

---

<!--  -->

## Create shared library from static library

**_References:_**

- [CMake org: [CMake] build a shared library from static libraries](https://cmake.org/pipermail/cmake/2008-March/020315.html)

### pkg-config

`pkg-config` will search `.pc` in `PKG_CONFIG_PATH`.

When you set `CMAKE_PREFIX_PATH` for the `.pc`, `pkg-config` still cannot find it. You need to add the `.pc` path into `PKG_CONFIG_PATH` via followings,

```makefile
set($ENV{PKG_CONFIG_PATH} <pc path>)
```

Ref [stackoverflow: pkg-config cannot find .pc files although they are in the path](https://stackoverflow.com/questions/11303730/pkg-config-cannot-find-pc-files-although-they-are-in-the-path)

**_References:_**

- [stackoverflow: set PKG_CONFIG_PATH in cmake](https://stackoverflow.com/a/44487317/4636081): The most voted answer' method is not correct.

<!--  -->
<br>

---

<!--  -->

### Set and Get Environment Variable

- Get: `$ENV{PKG_CONFIG_PATH}`

- Set: `set(ENV{PKG_CONFIG_PATH} <pc path>)`

**_References:_**

- [stackoverflow: set PKG_CONFIG_PATH in cmake](https://stackoverflow.com/a/44487317/4636081)
- [CMake doc: ENV](https://cmake.org/cmake/help/v3.14/variable/ENV.html)
- [CMake doc: Set Environment Variable](https://cmake.org/cmake/help/v3.14/command/set.html#set-environment-variable)

<!--  -->
<br>

---

<!--  -->

### Copy 3rdparty dynamic libs into the same folder as executable

```makefile
add_custom_command(TARGET <your_exe_name> POST_BUILD   # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy_if_different      # which executes "cmake - E copy_if_different..."
        "<your_3rdparty_lib>"                          # <--this is in-file
        $<TARGET_FILE_DIR:<your_exe_name>>)            # <--this is out-file path
```

Refer [stackoverflow: How to copy DLL files into the same folder as the executable using CMake?](https://stackoverflow.com/questions/10671916/how-to-copy-dll-files-into-the-same-folder-as-the-executable-using-cmake)

---

## Config compiler

```shell
CC=clang CXX=clang++ cmake ..
```

**_References::_**

- [stackoverflow: Switching between GCC and Clang/LLVM using CMake](https://stackoverflow.com/questions/7031126/switching-between-gcc-and-clang-llvm-using-cmake)

<!--  -->
<br>

---

<!--  -->

## Config with third party

### OpenMP

```makefile
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
```

**_Ref:_** [stackoverflow: How to set linker flags for OpenMP in CMake's try_compile function](https://stackoverflow.com/a/12404666/4636081)

<!--  -->
<br>

---

<br>
<!--  -->

# Pratice

## Demo cmake build opencv

```makefile
PROJECT(Test)

include_directories(/home/rivergold/software/lib/opencv/opencv-3.4.3/build/install/include)

#FIND_PACKAGE(OpenCV REQUIRED)

set(SRC_LIST main.cpp)

add_executable(Test ${SRC_LIST})

# Link your application with OpenCV libraries
target_link_libraries(Test libopencv_highgui.so libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)
```

**_References:_**

- [CSDN CMakeLists.txt 添加 opencv 库注意事项](https://blog.csdn.net/u012816621/article/details/51732932)

## Don't use `link_directories`

**Please use `find_library` and `target_link_libraries`**.

**_References:_**

- [stackoverflow: Cmake cannot find library using “link_directories”](https://stackoverflow.com/a/31471772/4636081)

<!--  -->
<br>

---

<br>
<!--  -->

# Error

## `undefined reference to xxx`

It is a link error. You need to use `TARGET_LINK_LIBRARIES` in cmakelist to add `.so` to your target.

# Concept

## `include(ExternalProject)`

**_References:_**

- [Blog: cmake 和其他构建工具协同使用](http://aicdg.com/oldblog/c++/2017/02/04/cmake-externalproject.html)

<!--  -->
<br>

---

<br>
<!--  -->

# Path

## CMAKE_SOURCE_DIR

Root project directory.

**理解:** 运行`cmake`所执行的最顶层的`CMakeLists.txt`所在的 directory

---

## CMAKE_CURRENT_LIST_DIR

When `.cmake`, use `CMAKE_CURRENT_LIST_DIR` to current `.cmake` directory.

**_References:_**

- [stackoverflow: In CMake, how can I find the directory of an included file?](https://stackoverflow.com/a/12854575/4636081)
