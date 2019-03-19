# 命令

## Copy file from host into container

In host terminal, run:

```bash
docker cp <host_file> <container_name>:<container_path>
```

Ref [stackoverflow: Copying files from host to Docker container](https://stackoverflow.com/questions/22907231/copying-files-from-host-to-docker-container)

## Show docker GUI on Linux

```bash
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix 
```

<!--  -->
<br>

***

<br>
<!--  -->

# 环境配置

## Install miniconda occur `Miniconda3-4.5.12-Linux-x86_64.sh: line 346: bunzip2: command not found`

```bash
yum install -y bzip2
```

Ref [CSDN: bunzip2: command not found](https://blog.csdn.net/zhao12501/article/details/79828964)

<!--  -->
<br>

***
<!--  -->

## [CentOS] Build cmake with gui

1. Download cmake source from [CMake](https://cmake.org/download/)

2. Decompress and cd

3. Run

```bash
# ./bootstrap --help
./bootstrap --qt-gui
make
make install
```

Ref [CMake: Installing CMake](https://cmake.org/install/)

If occur error like:

```bash
CMake Error at Modules/FindQt4.cmake:1314 (message):
  Found unsuitable Qt version "" from NOTFOUND, this code requires Qt 4.x
```

Run `yum install qt-devel` and try again.

<!--  -->
<br>

***
<!--  -->

## [Error] Run GUI in docker container, occur `cannot connect to X server :0`

On host, run:

```bash
xhost +
```
<!--  -->
<br>

***
<!--  -->

## [Error] Run `cmake-gui`, occur `X Error: BadDrawable (invalid Pixmap or Window parameter)`

In docker container, run:

```bash
export QT_X11_NO_MITSHM=1
cmake-gui
```

Ref [Github P0cL4bs/WiFi-Pumpkin: BadDrawable (invalid Pixmap or Window parameter) #53](https://github.com/P0cL4bs/WiFi-Pumpkin/issues/53#issuecomment-309120875)

<!--  -->
<br>

***
<!--  -->

# 第三方库

## OpenCV

### [Error] Build OpenCV-3.4.5 with CUDA-10.1 on CentOS-7.2, occur error

Error is followings:

```makefile
/opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:6676:95:   required from here
/opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:1067:16: error: cannot call member function 'void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]' without object
       __p->_M_set_sharable();
       ~~~~~~~~~^~
```

Ref [Nvidia Forums: Cuda 10.1: Nvidia, you're now "fixing" gcc bugs that gcc doesn't even have](https://devtalk.nvidia.com/default/topic/1048037/cuda-10-1-nvidia-you-re-now-quot-fixing-quot-gcc-bugs-that-gcc-doesn-t-even-have/?offset=10)

<!--  -->
<br>

***

<br>
<!--  -->
