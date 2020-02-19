# LLVM

# Install

I use `LLVM 8.0.1-rc1`.

## Download source

```bash
wget https://github.com/llvm/llvm-project/archive/llvmorg-8.0.1-rc1.zip
unzip xxx
```

## Build

```bash
cd llvm-project
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=./install  -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release ../llvm
```

***Ref:*** [LLVM doc: Getting Started Quickly (A Summary)](https://llvm.org/docs/GettingStarted.html#getting-started-quickly-a-summary)

### Error Memory or swap not enough

#### [Compile] Release or debug

Build llvm debug only you want to debug the compile. And when build llvm debug, it need much more memory.

Solution is [stackoverflow: Clang and LLVM - Release vs Debug builds](https://stackoverflow.com/questions/21643917/clang-and-llvm-release-vs-debug-builds)

#### Change swap size

```bash
# Count=n means you want n G
$ dd if=/dev/zero of=/swapfile1 bs=1G count=1
$ chmod 600 /swapfile1
$ mkswap /swapfile1
$ swapon /swapfile1
$ vi /etc/fstab
/swapfile1  swap  swap  defaults  0 0
```

Ref [2Day Geek: 3 Easy Ways To Create Or Extend Swap Space In Linux](https://www.2daygeek.com/add-extend-increase-swap-space-memory-file-partition-linux/#)

#### Delete swap file

```bash
swapoff -v <swap file name> # swapoff -v /swapfile
vim /etc/fstab
```

Ref [Redhat doc: 15.2.3. REMOVING A SWAP FILE](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/storage_administration_guide/swap-removing-file)

