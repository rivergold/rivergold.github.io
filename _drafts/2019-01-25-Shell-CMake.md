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

<br>

***

<br>

# CMake

## Common

### Path

- `PROJECT_SOURCE_DIR`: Project root path

- `CMAKE_CURRENT_SOURCE_DIR`: The path to the source directory currently being processed.

- `get_filename_component(PARENT_DIR ${MYPROJECT_DIR} DIRECTORY)`: Get parent path

    Refer [stackoverflow: CMake : parent directory ?](https://stackoverflow.com/questions/7035734/cmake-parent-directory)

- `CMAKE_BINARY_DIR`: The path to the top level of the build tree.
    ```bash
    mkdir build && cd build
    cmake ..
    # ${CMAKE_BINARY_DIR} is build/
    ```

### Tricks

#### Copy 3rdparty dynamic libs into the same folder as executable

```cmake
add_custom_command(TARGET <your_exe_name> POST_BUILD   # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy_if_different      # which executes "cmake - E copy_if_different..."
        "<your_3rdparty_lib>"                          # <--this is in-file
        $<TARGET_FILE_DIR:<your_exe_name>>)            # <--this is out-file path
```

Refer [stackoverflow: How to copy DLL files into the same folder as the executable using CMake?](https://stackoverflow.com/questions/10671916/how-to-copy-dll-files-into-the-same-folder-as-the-executable-using-cmake)