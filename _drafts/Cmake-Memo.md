# Cmake

# Common Variable

- **`PROJECT_SOURCE_DIR`**: Top level source directory for the current project.

    ```makefile
    project(<project name>)
    ```

- **`CMAKE_SOURCE_DIR`**: This is the full path to the top level of the current CMake source tree. For an in-source build, this would be the same as `CMAKE_BINARY_DIR`.

- **`CMAKE_CURRENT_SOURCE_DIR`**: The path to the source directory currently being processed.

- **`get_filename_component(PARENT_DIR ${MYPROJECT_DIR} DIRECTORY)`**: Get parent path

    ***Ref:*** [stackoverflow: CMake : parent directory ?](https://stackoverflow.com/questions/7035734/cmake-parent-directory)

- **`PROJECT_BINARY_DIR`**: Full path to build directory for project.

    - in-source: Top level source directory for the current project.
    - out-of-source: The directory of `cmake` command run.

    ***Ref:*** [CSDN: CMake PROJECT_BINARY_DIR和PROJECT_SOURCE_DIR区别](https://blog.csdn.net/sukhoi27smk/article/details/46388711)

<!-- - `CMAKE_BINARY_DIR`: The path to the top level of the build tree.
    ```bash
    mkdir build && cd build
    cmake ..
    # ${CMAKE_BINARY_DIR} is build/
    ``` -->

***References:***

- [维基教科书: CMake 入門/Out-of-source Build](https://zh.wikibooks.org/zh/CMake_%E5%85%A5%E9%96%80/Out-of-source_Build)

## &clubs; Defferent between `PROJECT_variable` and `CMAKE_variable`

`CMAKE_SOURCE_DIR` does indeed refer to the folder where the top-level CMakeLists.txt is defined.

However, `PROJECT_SOURCE_DIR` refers to the folder of the CMakeLists.txt containing the most recent project() command. When sub-project `CMakeLists.txt` has `prject(<project name>)`, `PROJECT_SOURCE_DIR` in sub-project will be changed into sub-project path. 

> This difference applies to all PROJECT_<var> vs CMAKE_<var> variables.

**理解:** `PROJECT_`是会基于`prject()`的，而`CMAKE_`不会。

***Ref:*** [stackoverflow: Are CMAKE_SOURCE_DIR and PROJECT_SOURCE_DIR the same in CMake?](https://stackoverflow.com/a/32030551/4636081)

<!--  -->
<br>

***

<br>
<!--  -->

# Common Command

## `CMAKE_PREFIX_PATH`

list of directories path to tell cmake where to search `find_package()`, `find_program()`, `find_library(), find_file()`

<!--  -->
<br>

***
<!--  -->

## `find_package()`

### &clubs; Use pkg-config to find package

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

***References:***

- [CMake doc: FindPkgConfig](https://cmake.org/cmake/help/v3.14/module/FindPkgConfig.html)


<!--  -->
<br>

***
<!--  -->

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

<!--  -->
<br>

***
<!--  -->

## `install`

***Ref:*** [博客园: CMake手册详情 install](https://www.cnblogs.com/coderfenghc/archive/2012/08/12/2627561.html)

***References:***

- [维基教科书: CMake 入門/輸出位置與安裝](https://zh.wikibooks.org/zh/CMake_%E5%85%A5%E9%96%80/%E8%BC%B8%E5%87%BA%E4%BD%8D%E7%BD%AE%E8%88%87%E5%AE%89%E8%A3%9D)

<!--  -->
<br>

***
<!--  -->

## `set`

<!--  -->
<br>

***

<br>
<!--  -->

# Tips

## The theory of find package

***Ref:*** [CSDN find_package与CMake如何查找链接库详解](https://blog.csdn.net/bytxl/article/details/50637277)

<!--  -->
<br>

***
<!--  -->

## Set cmake to find Python

`find_package( PythonInterp 3.5 REQUIRED )`

<!--  -->
<br>

***
<!--  -->

## Set cmake to find boost

```makefile
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS system) 
```

***References:***

- [stackoverflow: How do you add Boost libraries in CMakeLists.txt?](https://stackoverflow.com/a/6646518/4636081)

<!--  -->
<br>

***
<!--  -->

## Set cmake to find OpenCV

```makefile
find_package( OpenCV REQUIRED )
include_directories( $(OpenCV_INCLUDE_DIRS) )

target_link_libraries( <your target> ${OpenCV_LIBS})
```

***References:***

- [OpenCV doc: Using OpenCV with gcc and CMake](https://docs.opencv.org/3.0.0/db/df5/tutorial_linux_gcc_cmake.html)

<!--  -->
<br>

***
<!--  -->

## Create shared library from static library

***References:***

- [CMake org: [CMake] build a shared library from static libraries](https://cmake.org/pipermail/cmake/2008-March/020315.html)

### pkg-config

`pkg-config` will search `.pc` in `PKG_CONFIG_PATH`.

When you set `CMAKE_PREFIX_PATH` for the `.pc`, `pkg-config` still cannot find it. You need to add the `.pc` path into `PKG_CONFIG_PATH` via followings,

```makefile
set($ENV{PKG_CONFIG_PATH} <pc path>)
```

Ref [stackoverflow: pkg-config cannot find .pc files although they are in the path](https://stackoverflow.com/questions/11303730/pkg-config-cannot-find-pc-files-although-they-are-in-the-path)

***References:***

- [stackoverflow: set PKG_CONFIG_PATH in cmake](https://stackoverflow.com/a/44487317/4636081): The most voted answer' method is not correct.

<!--  -->
<br>

***
<!--  -->

### Set and Get Environment Variable

- Get: `$ENV{PKG_CONFIG_PATH}`

- Set: `set(ENV{PKG_CONFIG_PATH} <pc path>)`

***References:***

- [stackoverflow: set PKG_CONFIG_PATH in cmake](https://stackoverflow.com/a/44487317/4636081)
- [CMake doc: ENV](https://cmake.org/cmake/help/v3.14/variable/ENV.html)
- [CMake doc: Set Environment Variable](https://cmake.org/cmake/help/v3.14/command/set.html#set-environment-variable)

<!--  -->
<br>

***
<!--  -->

### Copy 3rdparty dynamic libs into the same folder as executable

```makefile
add_custom_command(TARGET <your_exe_name> POST_BUILD   # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy_if_different      # which executes "cmake - E copy_if_different..."
        "<your_3rdparty_lib>"                          # <--this is in-file
        $<TARGET_FILE_DIR:<your_exe_name>>)            # <--this is out-file path
```

Refer [stackoverflow: How to copy DLL files into the same folder as the executable using CMake?](https://stackoverflow.com/questions/10671916/how-to-copy-dll-files-into-the-same-folder-as-the-executable-using-cmake)

<!--  -->
<br>

***

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

***References:***

- [CSDN CMakeLists.txt添加opencv库注意事项](https://blog.csdn.net/u012816621/article/details/51732932)

## Don't use `link_directories`

**Please use `find_library` and `target_link_libraries`**.

***References:***

- [stackoverflow: Cmake cannot find library using “link_directories”](https://stackoverflow.com/a/31471772/4636081)

<!--  -->
<br>

***

<br>
<!--  -->

# Error

## `undefined reference to xxx`

It is a link error. You need to use `TARGET_LINK_LIBRARIES` in cmakelist to add `.so` to your target.