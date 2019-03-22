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

# Docker Config

## 改变存储位置

推荐方法：修改`/etc/docker/daemon.json`(没有该文件的话，还请创建)中的`data-root`

```json
{
    "runtimes": 
    {
        "nvidia": 
        {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "data-root": "/home/rivergold/software/docker"
}
```

Ref [Docker doc: Custom Docker daemon options](https://docs.docker.com/config/daemon/#custom-docker-daemon-options)

### 关于docker daemon的配置

有很多种方式对docker daemon进行设置:

- 官方推荐方式为修改`/etc/docker/daemon.json`

- CentOS中可能需要修改`/etc/sysconfig/docker` (本质上还是通过配置`system`)

***References:***

- [Docker Doc: Daemon configuration file](https://docs.docker.com/engine/reference/commandline/dockerd//#daemon-configuration-file)
- [Docker doc: Control Docker with systemd](https://docs.docker.com/config/daemon/systemd/)
- [CSDN: 通过systemd配置Docker](https://blog.csdn.net/xingwangc2014/article/details/50513946)
- [简书: docker daemon 配置文件](https://www.jianshu.com/p/2556a1c5d45d)
- [Blog Jin-Yang: systemd 使用简介](https://jin-yang.github.io/post/linux-systemd.html)
- [51CTO: 四个修改Docker默认存储位置的方法](https://blog.51cto.com/forangela/1949947)

## Docker Storage

devicemapper, overlay区别

- [Docker doc: Docker storage drivers](https://docs.docker.com/storage/storagedriver/select-storage-driver/)

### CentOS修改Container大小

该方法在公司的docker中测试是可以的

1. Edit `/etc/sysconfig/docker-storage`

2. Add `--storage-opt dm.basesize=30G`

    ```bash
    DOCKER_DEFAULT_STORAGE_OPTIONS='--data-root /data/docker -s devicemapper --storage-opt dm.use_def    erred_removal=true --storage-opt dm.use_deferred_deletion=true --storage-opt dm.basesize=30G'
    ```
3. Deletet all file in docker data root, get path via `docker info`. Make a backup of all your images

Ref [DOCKER COMMUNITY FORUMS: Increase container volume(disk) size](https://forums.docker.com/t/increase-container-volume-disk-size/1652/4)

***References:***

- [Blog hustcat: Docker内部存储结构（devicemapper）解析](http://hustcat.github.io/docker-devicemapper/)

<!--  -->
<br>

***

<br>
<!--  -->

# 使用docker部署环境

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

***Solution***

Use gcc 4.8.5 on CentOS 7

Ref [Nvidia Cuda doc: 1.1. System Requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)

<!--  -->
<br>

***
<!--  -->

### Build OpenCV from source

#### Need

- python-dev (include files and libs)
- FFMPEG

Ref [OpenCV doc: Installation in Linux](https://docs.opencv.org/3.4.5/d7/d9f/tutorial_linux_install.html)

【经验】：如果需要编译Python接口，需要注意的是：

- 安装Python-dev
    - [ ] 如果安装了anaconda，是不是就不用单独apt/yum install了？

- cmake时，需要check一下cmake的输出，看下是否找到了对应的Python

```bash
cmake -DWITH_CUDA=1 \
      -DCMAKE_INSTALL_PREFIX=/root/software/lib/opencv-3.4.5/build/install \
      -DOPENCV_PYTHON3_VERSION=3.6.8 \
      -DPYTHON3_EXECUTABLE=${ANACONDA_HOME}/envs/py3.6/bin/python3.6m  ..
```

#### Ubuntu

Ref [OpenCV doc: Installation in Linux](https://docs.opencv.org/3.4.5/d7/d9f/tutorial_linux_install.html)

Install ffmpeg， ref [stackoverflow: opencv Unable to stop the stream: Inappropriate ioctl for device](https://stackoverflow.com/a/45893821/4636081)

#### CentOS

1. Install `python-devel`

Ref [stackoverflow: How to install python developer package?](https://stackoverflow.com/questions/6230444/how-to-install-python-developer-package)

- [ ] 未完成

#### cmake时一些重要的显示信息

make

```bash
[100%] Linking CXX shared module ../../lib/python3/cv2.cpython-36m-x86_64-linux-gnu.so
[100%] Built target opencv_python3
```

make install

```bash
--   Python 3:
--     Interpreter:                 /root/software/anaconda/envs/py3.6/bin/python3.6m (ver 3.6.8)
--     Libraries:                   /root/software/anaconda/envs/py3.6/lib/libpython3.6m.so (ver 3.6.8)
--     numpy:                       /root/software/anaconda/envs/py3.6/lib/python3.6/site-packages/numpy/core/include (ver 1.16.2)
--     install path:                lib/python3.6/site-packages/cv2/python-3.6
```

#### Check OpenCV build information

- [Learn OpenCV: Get OpenCV Build Information ( getBuildInformation )](https://www.learnopencv.com/get-opencv-build-information-getbuildinformation/)

#### docker中编译好的OpenCV库在其他机器使用

1. 查看一下编译好的库需要哪些其他动态链接库

    ```bash
    $ readefl -d <cv2.cpython-36m-x86_64-linux-gnu.so>
    ```

2. 配置`/etc/ld.so.conf.d/`, 添加改库需要的其他动态库的路径

    ```bash
    $ ldconfig
    ```

3. Run

<!--  -->
<br>

***

<br>
<!--  -->
