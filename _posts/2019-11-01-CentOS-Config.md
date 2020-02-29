---
title: "CentOS Configuration"
last_modified_at: 2020-02-29
categories:
  - Memo
tags:
  - Linux
  - Tool
---

## :fallen_leaf:CentOS 源

Recommend using Ali source.

[阿里源](http://mirrors.aliyun.com/repo/)

```shell
wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
```

Update yum cache

```shell
yum clean all
yum makecache
```

**_References:_**

- [掘金: 「亲测有效」CentOS 解决 yum 命令出现 doesn't have enough cached 的问题](https://juejin.im/post/5d53d5ece51d4561e224a314)
- [清华源](https://mirror.tuna.tsinghua.edu.cn/help/centos/)

## :fallen_leaf:`gcc` Version Management

### What is SCL

**_Ref_** [运维之美: CentOS 下安装高版本 GCC](https://www.hi-linux.com/posts/25767.html)

### Use devtoolset to manage

<!-- 使用 devtoolset 管理 gcc 版本 -->

**_Ref:_** [Blog 源代码: 安装最新版 devtoolset-8](https://lrita.github.io/2018/11/28/upgrade-newest-devtoolset/)

### Install devtoolset

**_Ref:_** [Software Collections](https://www.softwarecollections.org/en/scls/?search=devtoolset&policy=&repo=&order_by=-create_date&per_page=10)

### Actiavte and Deactivate Devtoolset Permanently

add `source scl_source enable devtoolset-7` into your `~/.bashrc` or `~/.zshrc`

**_Ref_** [StackExchange: How to permanently enable scl CentOS 6.4?](https://unix.stackexchange.com/questions/175851/how-to-permanently-enable-scl-centos-6-4)

## :fallen_leaf:yum

### 安装 yum

Ref [博客园: CentOS yum 源的配置与使用](https://www.cnblogs.com/mchina/archive/2013/01/04/2842275.html)

- [ ] 还没有测试过

### Delete yum Repo

```shell
yum remove centos-release-scl
```

## :fallen_leaf:Develop Env Config

- zsh
- devtool
  - dev-toolset
  - llvm
- Python
- vim
- cmake
- git
- rzsz

### zsh

```bash
yum install zsh
# Set zsh as default shell
chsh -s $(which zsh)
# Install on-my-zsh
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
# Install zsh-syntax-highlighting
cd ~/.oh-my-zsh/plugins
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

Edit `~/.zshrc`

```bash
plugins=(
  git
  zsh-syntax-highlighting
  z)

# -----zsh
# Set *
setopt no_nomatch
# Forbiden rm -rf
alias rm='echo "rm is disabled, use trash or /bin/rm instead."'
# -----
```

**_Ref_** [知乎: mac 上使用 oh my zsh 有哪些必备的插件推荐?](https://www.zhihu.com/question/49284484/answer/617215298)

### Add yum Source Repo EPEL

```bash
yum install -y epel-release
```

**_Ref_** [博客园: Centos 解决 No package htop available](https://www.cnblogs.com/jiqing9006/p/10030886.html)

### Install Development Tools

```bash
yum group install "Development Tools"
```

**_Ref_** [nixCraft: RHEL / CentOS Linux Install Core Development Tools Automake, Gcc (C/C++), Perl, Python & Debuggers](https://www.cyberciti.biz/faq/centos-linux-install-gcc-c-c-compiler/)

#### Install Other gcc Version via `devtoolset`

```bash
yum install centos-release-scl
yum-config-manager --enable rhel-server-rhscl-7-rpms
yum install devtoolset-7
# Enable
scl enable devtoolset-7 bash
```

If you want to enable devtoolset as default, you need to add followings into your `~/.zshrc` or `~/.bashrc`

```shell
# Ubuntu
source scl_source enable devtoolset-8
# CentOS
source /opt/rh/devtoolset-7/enable
```

**_Ref_** [StackExchange: How to permanently enable scl CentOS 6.4?](https://unix.stackexchange.com/questions/175851/how-to-permanently-enable-scl-centos-6-4)

### LLVM

#### Install

```shell
yum install llvm-toolset-7
```

#### Enable

```shell
scl enable llvm-toolset-7 zsh

# vim ~/.zshrc add
source /opt/rh/llvm-toolset-7/enable
```

**_References:_**

- [stackoverflow: How to install Clang and LLVM 3.9 on CentOS 7](https://stackoverflow.com/questions/44219158/how-to-install-clang-and-llvm-3-9-on-centos-7)
- [Kuan-Yi Li's Blog: Installing Developer Toolset on RHEL-based Distributions](https://blog.abysm.org/2016/03/installing-developer-toolset-rhel-based-distributions/)

#### Compile from source.

```shell
cmake -DCMAKE_INSTALL_PREFIX=./install  -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release ../llvm
```

**_Ref_** [LLVM doc: Getting Started with the LLVM System](https://llvm.org/docs/GettingStarted.html#requirements)

##### Problem & Solutions

###### [Compile Error] `collect2: fatal error: ld terminated with signal 9 [Killed]`

You need to do one of the followings:

- Add more RAM to your VM, or
- Use gold instead of ld as a linker, or
- Build Release, not Debug build

Solution is [stackoverflow: llvm/clang compile error with Memory exhausted](https://stackoverflow.com/a/25221717)

##### [Compile] Release or debug

Build llvm debug only you want to debug the compile. And when build llvm debug, it need much more memory.

Solution is [stackoverflow: Clang and LLVM - Release vs Debug builds](https://stackoverflow.com/questions/21643917/clang-and-llvm-release-vs-debug-builds)

##### Change swap size

```shell
# Count=n means you want n G
$ dd if=/dev/zero of=/swapfile1 bs=1G count=1
$ chmod 600 /swapfile1
$ mkswap /swapfile1
$ swapon /swapfile1
$ vi /etc/fstab
/swapfile1  swap  swap  defaults  0 0
```

**_Ref_** [2Day Geek: 3 Easy Ways To Create Or Extend Swap Space In Linux](https://www.2daygeek.com/add-extend-increase-swap-space-memory-file-partition-linux/#)

##### Delete swap file

```shell
$ swapoff -v <swap file name> # swapoff -v /swapfile
$ vim /etc/fstab
```

**_Ref_** [Redhat doc: 15.2.3. REMOVING A SWAP FILE](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/storage_administration_guide/swap-removing-file)

### Python

1. Install **miniconda**
2. Config `~/.zshrc`

   ```bash
   export PATH=~/software/anaconda/bin:$PATH
   ```

### trash-cli

```bash
git clone https://github.com/andreafrancia/trash-cli.git
cd trash-cli
python setup.py install
```

**_Ref_** [Github andreafrancia/trash-cli](https://github.com/andreafrancia/trash-cli#from-sources)

### Vim

[This article]() introduces how to install Vim with Python3 supported.

### git

```shell
yum install http://opensource.wandisco.com/centos/7/git/x86_64/wandisco-git-release-7-2.noarch.rpm
yum install git
```

**_Ref_** [stackoverflow: How to install latest version of git on CentOS 7.x/6.x](https://stackoverflow.com/questions/21820715/how-to-install-latest-version-of-git-on-centos-7-x-6-x)

### rzsz

```bash
yum install lrzsz
```

**_Ref_** [CSDN: linux CentOS 安装 rz 和 sz 命令 lrzsz](https://blog.csdn.net/jack85986370/article/details/51321475)

## :fallen_leaf:Problems & Solutions

### Shell Cannot Show Chinese Character

After `yum update`, CentOS cannot show Chinese character, all Chinese character are question mark, and occur `Failed to set locale, defaulting to C` error when using `yum`

**Solution**

```shell
localedef -v -c -i en_US -f UTF-8 en_US.UTF-8
```

**_References:_**

- :thumbsup::thumbsup:[stackoverflow: locale-gen command in centos6](https://unix.stackexchange.com/a/140303)
- [CSDN: CentOS 下解决 ssh 登录 locale 警告](https://blog.csdn.net/Rainloving/article/details/69568618)
- [博客园: CentOS 下通过 locale 来设置字符集](https://www.cnblogs.com/pengdonglin137/p/3532615.html)

On Ubuntu, the command is `locale-gen en_US.UTF-8`

**_Ref_** [StackExchange-ask ubuntu: Ubuntu display Chinese Characters - Encoding Issue](https://askubuntu.com/questions/1070568/ubuntu-display-chinese-characters-encoding-issue)

Warning `/bin/sh: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)` in shell

**Solution:**

```shell
localedef -i en_US -f UTF-8 en_US.UTF-8
```

**_Ref_** [CSDN: /bin/sh: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)](https://blog.csdn.net/u013000139/article/details/81395140)
