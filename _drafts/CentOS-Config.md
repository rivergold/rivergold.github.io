# :fallen_leaf:Gcc Version Management

## What is SCL

**_Ref:_** [运维之美: CentOS 下安装高版本 GCC](https://www.hi-linux.com/posts/25767.html)

## Use devtoolset to manage

<!-- 使用 devtoolset 管理 gcc 版本 -->

Ref [Blog 源代码: 安装最新版 devtoolset-8](https://lrita.github.io/2018/11/28/upgrade-newest-devtoolset/)

### Install devtoolset

Ref [Software Collections](https://www.softwarecollections.org/en/scls/?search=devtoolset&policy=&repo=&order_by=-create_date&per_page=10)

### Actiavte and deactivate devtoolset permanently

add `source scl_source enable devtoolset-7` into your `~/.bashrc` or `~/.zshrc`

Ref [StackExchange: How to permanently enable scl CentOS 6.4?](https://unix.stackexchange.com/questions/175851/how-to-permanently-enable-scl-centos-6-4)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:yum

## 安装 yum

Ref [博客园: CentOS yum 源的配置与使用](https://www.cnblogs.com/mchina/archive/2013/01/04/2842275.html)

- [ ] 还没有测试过

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Develop Env Config

- zsh
- devtool
- Python
- vim
- cmake

## zsh

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

**_References:_**

- [知乎: mac 上使用 oh my zsh 有哪些必备的插件推荐?](https://www.zhihu.com/question/49284484/answer/617215298)

<!--  -->
<br>

---

<!--  -->

## Add yum source EPEL

```bash
yum install -y epel-release
```

Ref [博客园: Centos 解决 No package htop available](https://www.cnblogs.com/jiqing9006/p/10030886.html)

<!--  -->
<br>

---

<!--  -->

## Development Tools

```bash
yum group install "Development Tools"
```

Ref [nixCraft: RHEL / CentOS Linux Install Core Development Tools Automake, Gcc (C/C++), Perl, Python & Debuggers](https://www.cyberciti.biz/faq/centos-linux-install-gcc-c-c-compiler/)

### Install other gcc version

```bash
yum install centos-release-scl
yum-config-manager --enable rhel-server-rhscl-7-rpms
yum install devtoolset-7
# Enable
scl enable devtoolset-7 bash
```

If you want to enable devtool as default, you need to add followings into your `~/.zshrc` or `~/.bashrc`

```shell
source scl_source enable devtoolset-8
```

Ref [StackExchange: How to permanently enable scl CentOS 6.4?](https://unix.stackexchange.com/questions/175851/how-to-permanently-enable-scl-centos-6-4)

<!--  -->
<br>

---

<!--  -->

## Python

1. Install **miniconda**
2. Config `~/.zshrc`

   ```bash
   export PATH=~/software/anaconda/bin:$PATH
   ```

<!--  -->
<br>

---

<!--  -->

## trash-cli

```bash
git clone https://github.com/andreafrancia/trash-cli.git
cd trash-cli
python setup.py install
```

Ref [Github andreafrancia/trash-cli](https://github.com/andreafrancia/trash-cli#from-sources)

<!--  -->
<br>

---

<!--  -->

## LLVM

Compile from source.

```bash
cmake -DCMAKE_INSTALL_PREFIX=./install  -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release ../llvm
```

Ref [LLVM doc: Getting Started with the LLVM System](https://llvm.org/docs/GettingStarted.html#requirements)

### Problem & Solutions

#### [Compile Error] `collect2: fatal error: ld terminated with signal 9 [Killed]`

You need to do one of the followings:

- Add more RAM to your VM, or
- Use gold instead of ld as a linker, or
- Build Release, not Debug build

Solution is [stackoverflow: llvm/clang compile error with Memory exhausted](https://stackoverflow.com/a/25221717)

### [Compile] Release or debug

Build llvm debug only you want to debug the compile. And when build llvm debug, it need much more memory.

Solution is [stackoverflow: Clang and LLVM - Release vs Debug builds](https://stackoverflow.com/questions/21643917/clang-and-llvm-release-vs-debug-builds)

## Change swap size

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

## Delete swap file

```bash
$ swapoff -v <swap file name> # swapoff -v /swapfile
$ vim /etc/fstab
```

Ref [Redhat doc: 15.2.3. REMOVING A SWAP FILE](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/storage_administration_guide/swap-removing-file)

<!--  -->
<br>

---

<!--  -->

## vim

### Install dependence

```bash
yum install -y ruby ruby-devel lua lua-devel luajit \
luajit-devel ctags git python python-devel \
tcl-devel \
perl perl-devel perl-ExtUtils-ParseXS \
perl-ExtUtils-XSpp perl-ExtUtils-CBuilder \
perl-ExtUtils-Embed
```

### Build (with Python3)

```bash
./configure  --with-features=huge \
             --enable-multibyte \
             --enable-rubyinterp=yes \
             --enable-pythoninterp=yes \
             --with-python-config-dir=/usr/lib/python2.7/config \
             --enable-python3interp=yes \
             --with-python3-config-dir=/root/software/anaconda/lib/python3.7/config-3.7m-x86_64-linux-gnu \
             --enable-perlinterp=yes \
             --enable-luainterp=yes \
             --enable-gui=gtk2 \
             --enable-cscope \
             --prefix=/usr/local
```

Then run `make` and `make install`.

Run `vim --version` to check if vim support Python, like `+python3`. And run followings to check if vim can run with python.

```vim
$ vim
:py3 pass
```

#### [Error] Run `:py3 pass` occur: Cannot load python3.7m.a

我是用的是 Anaconda 安装的 Python，其 python-config 在`${AnacondaHome}/lib/python3.7/config-3.7m-x86_64-linux-gnu/`. 在进行`./configure`时，显示 vim 链接的 python 库为`libpython3.7m.a`（`checking Python3's dll name... libpython3.7m.a`）, 由于该库为静态库，导致 vim 在启动时，无法加载 python 的动态库，从而导致失败，也无法成功使用 Youcompleteme。

解决方法为：修改执行 vim 的`./configure`后的`src/auto/config.mk`，让其链接对应的动态库。需要稍微注意的点是如果`${AnacondaHome}/lib/python3.7/config-3.7m-x86_64-linux-gnu/`中没有`libpython3.7m.so`，还需要将该库软连接到该目录。

```vim
PYTHON3_CFLAGS  = -I/root/software/anaconda/include/python3.7m -pthread -DDYNAMIC_PYTHON3_DLL=\"libpython3.7m.so\"
```

由于 vim 开启时还需要动态加载 python 动态库，所以还需要将`libpython3.7m.so`添加到系统库的路径中。我采用的方法是：在`/etc/ld.so.conf.d/`中创建动态库路径，之后运行`ldconfig`使其生效。

Ref [Github vim/vim: Build from source linking to Anaconda python fails #2549](https://github.com/vim/vim/issues/2549#issuecomment-358535165)

**_References:_**

- [stackoverflow: VIM installation and Anaconda](https://stackoverflow.com/questions/29219553/vim-installation-and-anaconda)

### Config

Edit `~/.vimrc`

```vim
set nocompatible              " be iMproved, required
filetype off                  " required
set nu!
syntax on
set ts=4
set backspace=2
set expandtab

set encoding=utf-8  " The encoding displayed.
set fileencoding=utf-8  " The encoding written to file.
```

### Plugin

```bash
mkdir -p ~/.vim/bundle
cd ~/.vim/bundle
git clone <plugin>
# Follow plugin git readme to build and install
# ...
```

Ref [howchoo.com: How to install Vim plugins without a plugin manager](https://howchoo.com/g/ztmyntqzntm/how-to-install-vim-plugins-without-a-plugin-manager)

#### Using vim-pathogen to manage runtimepath to load plugin

Ref [Github: tpope/vim-pathogen](https://github.com/tpope/vim-pathogen)

```bash
mkdir -p ~/.vim/autoload ~/.vim/bundle && \
curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim
```

Add followings into `.vimrc`

```vim
execute pathogen#infect()
```

#### Plugin needed:

- [YouCompleteMe](https://github.com/Valloric/YouCompleteMe)

  ```bash
  python setup.py --clang-completer --system-libclang
  ```

  **_References:_**

  - [Github Valloric/YouCompleteMe: CMake Error at ycm/CMakeLists file DOWNLOAD HASH mismatch #1711](https://github.com/Valloric/YouCompleteMe/issues/1711#issuecomment-329520570)
