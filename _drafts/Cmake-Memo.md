# Cmake

## Basics

- `CMAKE_PREFIX_PATH` list of directories path to tell cmake where to search `find_package()`, `find_program()`, `find_library(), find_file()`

- `find_package()`

<br>

***

<br>

## Tips

***References:***

- [CSDN find_package与CMake如何查找链接库详解](https://blog.csdn.net/bytxl/article/details/50637277)

### Set cmake to find Python

`find_package( PythonInterp 3.5 REQUIRED )`

### Set cmake to find boost

```makefile
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS system) 
```

***References:***

- [stackoverflow: How do you add Boost libraries in CMakeLists.txt?](https://stackoverflow.com/a/6646518/4636081)

### Set cmake to find OpenCV

```makefile
find_package( OpenCV REQUIRED )
include_directories( $(OpenCV_INCLUDE_DIRS) )

target_link_libraries( <your target> ${OpenCV_LIBS})
```

***References:***

- [OpenCV doc: Using OpenCV with gcc and CMake](https://docs.opencv.org/3.0.0/db/df5/tutorial_linux_gcc_cmake.html)

<br>

***

<br>

## Pratice

### Demo cmake build opencv

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

### Don't use `link_directories`

**Please use `find_library` and `target_link_libraries`**.

***References:***

- [stackoverflow: Cmake cannot find library using “link_directories”](https://stackoverflow.com/a/31471772/4636081)

## Make Error

### `undefined reference to xxx`

It is a link error. You need to use `TARGET_LINK_LIBRARIES` in cmakelist to add `.so` to your target.