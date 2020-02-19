## Build Caffe

It is recommended to build Caffe using `Makefile.config`. I still cannot build via Camke. 

Another thing needed to be noted is caffe `Makefile.config` cannot find `INCLUDE_DIRS` and `LIBRARY_DIRS` via os setting. A typical example is I create `boost.conf` in `/etc/ld.so.conf.d` but `Makefile.config` still cannot find boost.

### Build on CentOS 7.2

Mainly reference [this blog](https://www.mtyun.com/library/how-to-install-caffe-on-centos7).

#### Build and install boost from source

Because `yum` boost package is too old and not support Python3, so it's better to build it from source.

***References:***

- [Blog: boost python3依赖安装](https://www.cnblogs.com/freeweb/p/9209362.html)

## Problems & Solutions

### Error when build `caffe` with `cpu_only=1`: `In file included from src/caffe/util/interp.cpp:4:0: ./include/caffe/util/interp.hpp:6:23: fatal error: cublas_v2.h: No such file or directory #include <cublas_v2.h>`

Edit `./include/caffe/util/interp.hpp`

```c++
./include/caffe/util/interp.hpp
```

to

```c++
#ifdef CPU_ONLY
#else
#include <cublas_v2.h>
#endif
```

***References:***

- [stackoverflow: Installing Caffe without CUDA: fatal error: cublas_v2.h No such file (Fedora23)](https://stackoverflow.com/a/35286011/4636081)

### Eroor when build `caffe` with `atlas`, occur error:

```makefile
/usr/bin/ld: cannot find -lcblas
/usr/bin/ld: cannot find -latlas
collect2: error: ld returned 1 exit status
```

Edit `Makefile.config` and add `/usr/lib64/atlas` into `LIBRARY_DIRS`

```makefile
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib64/atlas
```

### Link error (cannot find xxx.so) when run `make runtest`

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your path>
make runtest
unset LD_LIBRARY_PATH
```

***References:***

- [Google Groups: Error during compilation (make runtest): "error while loading shared libraries: libcudart.so.6.5"](https://groups.google.com/d/msg/caffe-users/dcZrE3-60mc/XYQRIDrmBgAJ)

## Tips

### Python load caffe model

***References:***

- [stackoverflow: How do I load a caffe model and convert to a numpy array?](https://stackoverflow.com/a/45208380/4636081)