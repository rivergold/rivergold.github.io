# Cmake

## Basics

- `CMAKE_PREFIX_PATH` list of directories path to tell cmake where to search `find_package()`, `find_program()`, `find_library(), find_file()`

## Tips

***References:***

- [CSDN find_package与CMake如何查找链接库详解](https://blog.csdn.net/bytxl/article/details/50637277)

### Set cmake to find Python

`find_package( PythonInterp 3.5 REQUIRED )`

## Pratice

### Demo cmake build opencv

```makefile
PROJECT(Test)

include_directories(/home/rivergold/software/lib/opencv/opencv-3.4.3/build/install/include)
link_directories(/home/rivergold/software/lib/opencv/opencv-3.4.3/build/install/lib)

#FIND_PACKAGE(OpenCV REQUIRED)

set(SRC_LIST main.cpp)

add_executable(Test ${SRC_LIST})

# Link your application with OpenCV libraries
target_link_libraries(Test libopencv_highgui.so libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)
```

***References:***

- [CSDN CMakeLists.txt添加opencv库注意事项](https://blog.csdn.net/u012816621/article/details/51732932)