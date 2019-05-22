# Shell

## Common

### Check if the file exist or not

```bash
#!/bin/bash
file=./file
if [ -e "$file" ]; then
    echo "File exists"
else
    echo "File does not exist"
fi
```

Refer [stackoverflow: How do I tell if a regular file does not exist in Bash?][stackoverflow: How do I tell if a regular file does not exist in Bash?]

[stackoverflow: How do I tell if a regular file does not exist in Bash?]: https://stackoverflow.com/questions/638975/how-do-i-tell-if-a-regular-file-does-not-exist-in-bash

### Params

***References:***

- [IBM Developer: Bash 参数和参数扩展](https://www.ibm.com/developerworks/cn/linux/l-bash-parameters.html)

<!--  -->
<br>

***

<br>
<!--  -->

# CMake

## Common Variable

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

### &clubs; Defferent between `PROJECT_variable` and `CMAKE_variable`

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

## Find Package

### Use pkg-config to find package

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

## Install

***Ref:*** [博客园: CMake手册详情 install](https://www.cnblogs.com/coderfenghc/archive/2012/08/12/2627561.html)

***References:***

- [维基教科书: CMake 入門/輸出位置與安裝](https://zh.wikibooks.org/zh/CMake_%E5%85%A5%E9%96%80/%E8%BC%B8%E5%87%BA%E4%BD%8D%E7%BD%AE%E8%88%87%E5%AE%89%E8%A3%9D)

<!--  -->
<br>

***

<br>
<!--  -->

## Tricks

## pkg-config

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