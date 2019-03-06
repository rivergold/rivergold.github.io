# Deep Learning on Mobile Devices

**Improtant: You'd beeter write your code in C++, because it's easy to port.**

# Caffe2

**Note: PyTorch(>=1.0) has integrated caffe2, and you can build caffe2 in PyTorch source code.**

PyTorch use onnx -> caffe2 to run on mobile devices.

(caffe2 C++ api doc is not complete yet)

## Good websites

- [Github leonardvandriel/caffe2_cpp_tutorial][caffe2_cpp_tutorial]: Caffe2 C++ api examples

[caffe2_cpp_tutorial]: https://github.com/leonardvandriel/caffe2_cpp_tutorial

## Build

### Android

1. Build caffe2 for android using `pytorch/scripts/build_android.sh`

    **Note:** Change `gcc` to `clang`
2. Set CMakeLists to contain **c10**, **caffe2**, **ATen** and **google** include files.

    More details please look at the [AICamera_new](https://github.com/wangnamu/AICamera_new) project.

    **Note:** The tensor value setting in this project is incorrect. The right way is from [Github/caffe2_cpp_tutorial][caffe2_cpp_tutorial].

***References:***

- [Gtihub pytorch/pytorch: [Caffe2] Caffe2 for Android has no include files #14353](https://github.com/pytorch/pytorch/issues/14353)
- [Gtihub wangnamu/AICamera_new](https://github.com/wangnamu/AICamera_new)
- [Github leonardvandriel/caffe2_cpp_tutorial](https://github.com/leonardvandriel/caffe2_cpp_tutorial)

<br>

***

<br>

# TFLite

## Android

### Problems & Solutions

#### [Compile] Error run `interpreter->ResizeInputTensor(input, size)`

***References:***

- [腾讯云: tensorflow lite（tflite）在调整输入demonsion之后调用错误](https://cloud.tencent.com/developer/ask/200429)

#### [Compile] Cannot set value to model

Using `tflite_interpreter_ptr_->inputs().size()`, the input size is not fixed and will occur vary large number.

***Solution:***

- [Github tensorflow/tensorflow: TensorFlowlite 加载模型报 it is probably compressed](https://github.com/tensorflow/tensorflow/issues/22333)

#### [Compile] Cannot find `flatbuffers/flatbuffers.h` head file

Have a look at `third_party/flatbuffers/workspace.bzl`, download flatbuffer from the link and copy the include folder as flatbuffers include.

***References:***

- [Github tensorflow/tensorflow: cannot find "flatbuffers/flatbuffers.h" head file #21965](https://github.com/tensorflow/tensorflow/issues/21965)

#### [Compile] Result when set `interpreter->UseNNAPI(true)` is different

When set `true`: result is right; when set `false`, result is very big and not correct.

***Solution:***

- [ ] TODO

<br>

***

<br>

# Build 3rdparty for Android

- [Build]: Build from 3rdparty source;
- [Compile]: Compile with 3rdparty api;
- [Link]: Complie link 3rdparty 

## OpenCV

**Tips: Better use android sdk <= 23 to build opencv for android.**

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=<ndk/android.toolchain.cmake> \
      -DANDROID_ABI="armeabi-v7a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-23 \
      -DCMAKE_INSTALL_PREFIX=<install path> \
      -DBUILD_JAVA=OFF -DBUILD_ANDROID_EXAMPLES=OFF ..
```

***References:***

- [stackoverflow: NDK - problems after GNUSTL has been removed from the NDK (revision r18)](https://stackoverflow.com/questions/52410712/ndk-problems-after-gnustl-has-been-removed-from-the-ndk-revision-r18/52436751#52436751)
- [Github opencv/opencv: SDK Tools, Revision 25.3.0 and OpenCVDetectAndroidSDK.cmake #8460](https://github.com/opencv/opencv/issues/8460#issuecomment-418232967): Solution for error `downgrade your Android sdk to 25.3.0`

### Problems & Solutions

#### [Building] Error about `error: unknown directive .func png_read_filter_row_sub4_neon`

***Solution:***

```bash
sudo apt install gcc-arm-linux-gnueabi
```

- [Linux公社: 解决一个Ubuntu中编译NEON优化的OpenCV的错误](https://www.linuxidc.com/Linux/2018-09/154272.htm)

#### [Link] Error `Android Studio with NDK : link error : undefined reference to 'stderr'`

***Solution:***

Using android sdk version <= 23.

- [stackoverflow: Android Studio with NDK : link error : undefined reference to 'stderr'](https://stackoverflow.com/questions/51767214/android-studio-with-ndk-link-error-undefined-reference-to-stderr)

#### [NDK] When build opencv with `-DANDROID_STL=c++_static`, then use built opencv to build another lib with link with `c++_static` occur error like followings. But when link with `c++_shared`, no error occurs.

```bash
/home/rivergold/iqiyi/Project/Rivergold/OpenCV-Android/opencv-3.4.5/modules/core/src/logger.cpp:0: error: undefined reference to 'std::__ndk1::cerr'
/home/rivergold/iqiyi/Project/Rivergold/OpenCV-Android/opencv-3.4.5/modules/core/src/logger.cpp:0: error: undefined reference to 'std::__ndk1::cout'
/home/rivergold/iqiyi/Project/Rivergold/OpenCV-Android/opencv-3.4.5/modules/core/src/logger.cpp:0: error: undefined reference to 'std::__ndk1::cerr'
/home/rivergold/software/Android-SDK/android-ndk-r19-linux-x86_64/android-ndk-r19/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/c++/v1/ios:0: error: undefined reference to 'std::__ndk1::cerr'
CMakeFiles/face_landmarks_detector.dir/mtcnn.cpp.o:mtcnn.cpp:function MTCNN::refine(std::__ndk1::vector<Bbox, std::__ndk1::allocator<Bbox> >&, int const&, int const&, bool): error: undefined reference to 'std::__ndk1::cout'
clang++: error: linker command failed with exit code 1 (use -v to see invocation)
CMakeFiles/face_landmarks_detector.dir/build.make:162: recipe for target 'libface_landmarks_detector.so' failed
make[2]: *** [libface_landmarks_detector.so] Error 1
CMakeFiles/Makefile2:67: recipe for target 'CMakeFiles/face_landmarks_detector.dir/all' failed
make[1]: *** [CMakeFiles/face_landmarks_detector.dir/all] Error 2
Makefile:83: recipe for target 'all' failed
make: *** [all] Error 2
```

I think there are some things wrong in NDK `libstdc++.a`.

**Note:** `libstdc++.a` path is `<ndk>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/arm-linux-androideabi/<SDK-Version>/libc++.so`


<br>

***

<br>

# Tricks

## Move your code into android

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-092e5079a7724bbb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="80%">
</p>

**Very important:** In JNI, you must manage C++ memory yourself. It means when you new a memory area (using `new`), you must remember to delete it.

```C++
unsigned char *buf = new unsigned char[len];
// ... do process
delete buf;
```

<br>

***

<br>

# Android开发的坑

## Tips & Tricks

### Load model file into sdk card

### Memory profiler

### CMake clean

Manual delete `.externalNativeBuild`

<br>

***

<br>

# TVM

## Basic Concepts

### LLVM

***References:***

- [Linux中国: 为什么人人都该懂点LLVM](https://linux.cn/article-6073-1.html)
- [知乎: 请问LLVM与GCC之间的关系，网上说LLVM 是编译器的架构，在这个架构上可以搭建多个小编译器（类似C、C++/JAVA/)，不知理解的对不对，还请高手补充？](https://www.zhihu.com/question/20039402)
- [掘金: 《深入理解 LLVM》第一章 LLVM 简介](https://juejin.im/entry/5874d80761ff4b006d546b2f)

### Framework

- NNVM: Does the graph-level optimization
- TVM: Does the tensor-level optimization

Refer [TVM-tutorials: nnvm_quick_start.py](https://github.com/dmlc/tvm/blob/881a78b3d6fc092e2c1477ecf37868382b501684/tutorials/nnvm_quick_start.py#L67).

### Devices

#### ROCM

AMD's "CUDA"

***References:***

- [CSDN: AMD ROCm 平台简介](https://blog.csdn.net/JackyTintin/article/details/74637157)

<br>

***

<br>

## TFLite to TVM

***References:***

- [知乎-蓝色专栏: 使用TVM支持TFLite（上）](https://zhuanlan.zhihu.com/p/55136595)
- [知乎-蓝色专栏: 使用TVM支持TFLite（中）](https://zhuanlan.zhihu.com/p/55583443)
- [TVM Docs-Tutorials: Compile TFLite Models](https://docs.tvm.ai/tutorials/frontend/from_tflite.html#sphx-glr-tutorials-frontend-from-tflite-py)