# Build and Install

**If you want to use vim with anaconda Python well, strongly suggest to build vim from source.**

Download src from [Github](https://github.com/vim/vim/releases).

Build vim need:

- Python3 and python3-dev

**Note:**

- Use system default gcc

## Ubuntu

### 1. Install `python-dev`\*\*

```shell
sudo apt-get install python3-dev
# Pay attention to which your anaconda python version is.
```

### 2. run Config\*\*

Use `./configure --help` to get command list

**Latest vim version**

```shell
./configure --with-features=huge \
--enable-multibyte \
--enable-rubyinterp=yes \
--enable-python3interp=yes \
--with-python3-command=/root/software/anaconda/bin/python \
--enable-perlinterp=yes \
--enable-luainterp=yes \
--enable-gui=gtk2 \
--enable-cscope \
--prefix=/usr/local \
--enable-fail-if-missing
```

**Older version**

```shell
./configure --with-features=huge \
--enable-multibyte \
--enable-rubyinterp=yes \
# --enable-pythoninterp=yes \
# --with-python-config-dir=/usr/lib/python2.7/config \
--enable-python3interp=yes \
--with-python3-config-dir=~/software/anaconda/bin/ \
--enable-perlinterp=yes \
--enable-luainterp=yes \
--enable-gui=gtk2 \
--enable-cscope \
--prefix=/usr/local \
--enable-fail-if-missing
```

When config vim, need `python-config` for Python2 and `python3-config` for Python3. You can run `python3-config --configdir` to get path of it.

**Note-1:** I would also recommend running configure with --enable-fail-if-missing so that the configure script will fail instead of quietly warning that it didn't find a python3 config directory or executable.

**Note-2:** vim can be built with python2 and python3, but when install `Youcompleteme`, it need only one python version. So, when you build vim, you'd better only choose on python version. If you want to build vim with python2, you need to change `--enable-python3interp=yes` to `--enable-pythoninterp=yes` and `--with-python3-config-dir=~/software/anaconda/bin/` to `--with-python-config-dir=/usr/lib/python2.7/config`

**_References:_**

- [stackoverflow: VIM installation and Anaconda](https://stackoverflow.com/a/41917764/4636081)

- [vim - configure for python3 support but still shows -python3 in version information](https://stackoverflow.com/a/26443517/4636081)

### 3. build

```shell
make -j8
sudo make install
```

---

## CentOS

### 1. Install dependences

```shell
yum install -y ruby ruby-devel lua lua-devel luajit \
luajit-devel ctags git python python-devel \
tcl-devel \
perl perl-devel perl-ExtUtils-ParseXS \
perl-ExtUtils-XSpp perl-ExtUtils-CBuilder \
perl-ExtUtils-Embed
```

**Install `python3-dev`**:

```shell
yum search python3 | grep devel
yum install python3-dev
```

**_References:_**

- [stackoverflow: How to install python3-devel on red hat 7](https://stackoverflow.com/questions/43047284/how-to-install-python3-devel-on-red-hat-7)

#### **[Problems]**

Centos 7.2 install `python36-devel` occur error `Requires: libcrypto.so.10(openssl.1.0.2)(64bit)`

**Solution**

You need to update openssl from `1.0.1` to `1.0.2`. You nend download [openssl-libs-1.0.2k-12.el7.x86_64.rpm](https://centos.pkgs.org/7/centos-x86_64/openssl-libs-1.0.2k-12.el7.x86_64.rpm.html) and run `rpm -i openssl-libs-1.0.2k-12.el7.x86_64.rpm`. And then it may occur another error like `conflicts with file from package`, you can use `rpm -i --replacefiles openssl-libs-1.0.2k-12.el7.x86_64.rpm`

**_References:_**

- [Blog: 解决 CentOS 下的 conflicts with file from 错误.](http://rayfuxk.iteye.com/blog/2280643)

### 2. Config

Same with Ubuntu

### 3. Build

Same with Ubuntu

---

## Build Problems & Solutions

### [Configure Error] `no terminal library found`

**Ubuntu**

```shell
sudo apt install libncurses5-dev
```

**CentOS**

```shell
yum install ncurses-devel.x86_64
```

**_References:_**

- [CSDN: CentOS 编译 vim no terminal library found](https://blog.csdn.net/cuijianzhi/article/details/78652745)

---

## Test

1. Run `vim --version` to check if vim support Python3, like `+python3`

2. Run `vim` and input `:py3 pass`

### Problems & Solutions

#### [Error] Run `:py3 pass` occur: Cannot load python3.7m.a

我是用的是 Anaconda 安装的 Python，其 python-config 在`${AnacondaHome}/lib/python3.7/config-3.7m-x86_64-linux-gnu/`. 在进行`./configure`时，显示 vim 链接的 python 库为`libpython3.7m.a`（`checking Python3's dll name... libpython3.7m.a`）, 由于该库为静态库，导致 vim 在启动时，无法加载 python 的动态库，从而导致失败，也无法成功使用 Youcompleteme。

解决方法为：修改执行 vim 的`./configure`后的`src/auto/config.mk`，让其链接对应的动态库。需要稍微注意的点是如果`${AnacondaHome}/lib/python3.7/config-3.7m-x86_64-linux-gnu/`中没有`libpython3.7m.so`，还需要将该库软连接到该目录。

```vim
PYTHON3_CFLAGS  = -I/root/software/anaconda/include/python3.7m -pthread -DDYNAMIC_PYTHON3_DLL=\"libpython3.7m.so\"
```

由于 vim 开启时还需要动态加载 python 动态库，所以还需要将`libpython3.7m.so`添加到系统库的路径中。我采用的方法是：在`/etc/ld.so.conf.d/`中创建动态库路径，之后运行`ldconfig`使其生效。

**_Ref:_** [Github vim/vim: Build from source linking to Anaconda python fails #2549](https://github.com/vim/vim/issues/2549#issuecomment-358535165)

**_References:_**

- [stackoverflow: VIM installation and Anaconda](https://stackoverflow.com/questions/29219553/vim-installation-and-anaconda)

<!--  -->
<br>

---

<br>
<!--  -->

# Config

Strongly suggest to use **[The Ultimate vimrc](https://github.com/amix/vimrc)**.

**Note:** You maybe say that SpaceVim is good, but I think it is beautiful but is slow when using.

## Install `The Ultimate vimrc`

```bash
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh
```

## Config vim

Just edit vim `~/.vim_runtime/my_configs.vim`

## Common Use

**The leader is `,`**

<!--  -->
<br>

---

<br>
<!--  -->

# Plugin

**The Ultimate vimrc** install vim **pathogen**.

```bash
cd ~/.vim_runtime
git clone <the plugin> # Follow the plugin installation
```

## gruvbox

- gruvbox

```bash
git clone https://github.com/morhetz/gruvbox.git ~/.vim_runtime/
```

Edit `my_configs.vim`

```vim
colorscheme gruvbox
```

**_Ref:_** [Github morhetz/gruvbox: Installation](https://github.com/morhetz/gruvbox/wiki/Installation)

<!--  -->
<br>

---

<!--  -->

## YouCompleteMe

### Requirements

- vim with python2 and python3 support
- llvm >= 7.0.0
- Raw Python (not Ananconda Python)

### Install

```bash
cd ~/.vim_runtime/my_plugins
git clone --recursive https://github.com/Valloric/YouCompleteMe.git
```

### Compile

```bash
cd YouCompleteMe
# Use system Python (not Ananconda Python)
/usr/bin/python3.7m install.py --clang-com --system-libclang
```

**Why use `--system-libclang`:** [Gtihub Valloric/YouCompleteMe CMake Error at ycm/CMakeLists file DOWNLOAD HASH mismatch #1711](https://github.com/Valloric/YouCompleteMe/issues/1711#issuecomment-329520570)

### Use

#### Navigate in complete box

- `ctr` + `n`: down in complete box

- `ctr` + `p`: up in complete box

**_References:_**

- [stackoverflow: Vim-style keys to navigate up and down in Omnicomplete box](https://stackoverflow.com/questions/21900031/vim-style-keys-to-navigate-up-and-down-in-omnicomplete-box)

### Problem & Solution

#### [Error] `import ycm_core as ycm_core`

1. `cd YouCompleteMe/third_party/ycmd`
2. `ldd ./ycm_core.so` to check which dynamic library is not work

**_Referneces:_**

- [Youcomplete 完全安装](https://my.oschina.net/pointeraddress/blog/855916)

<!--  -->
<br>

---

<!--  -->

## NERD Tree

Default installed with **The Ultimate vimrc**.

```bash
map <leader>nn :NERDTreeToggle<cr>
map <leader>nb :NERDTreeFromBookmark
map <leader>nf :NERDTreeFind<cr>
```

### Shortcut

- Hold `ctr`, `w` + `w`: switch between edit window and tree window

**_References:_**

- [简书: Vim 插件--Nerdtree 的使用及快捷键](https://www.jianshu.com/p/e58d92c65695)

<!--  -->
<br>

---

<br>
<!--  -->

# My `my_configs.vim`

```vim
colorscheme gruvbox " Color theme gruvbox

" -----basic config
syntax on " 开启语法高亮
set number " 显示行号
set ts=4 " 设置tab为4个空格
" -----

" -----Disable python space error
if exists('python_highlight_all')
    unlet python_highlight_all
endif
if exists('python_space_error_highlight')
    unlet python_space_error_highlight
endif
" -----

" -----Shortcut
:cmap qq q!
" open number
:cmap <leader>n set nu
:cmap <leader>cn set nu!
```

<!--  -->
<br>

---

<br>
<!--  -->

# Tips

## Reload file

`:e`

**_Ref:_** [stackoverflow: refresh changed content of file opened in vi(m)](https://unix.stackexchange.com/questions/149209/refresh-changed-content-of-file-opened-in-vim)

<!--  -->
<br>

---

<br>
<!--  -->

# Problems & Solutions

## YouCompleteMe

### [Run] When run `vim` occur `The ycmd server SHUT DOWN (restart with ':YcmRestartServer'). Unexpected error while loading the YCM core library. Type ":YcmToggleLogs ycmd_58963_stderr_d1zdkhog.log" to check the logs.`

After run `:YcmToggleLogs ycmd_58963_stderr_d1zdkhog.log`, occur followings:

```bash
 Traceback (most recent call last):
   File "/home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycmd/utils.py", line 637, in ImportAndCheckCore
     ycm_core = ImportCore()
   File "/home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycmd/utils.py", line 628, in ImportCore
     import ycm_core as ycm_core
 ImportError: /home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycm_core.so: undefined symbol:                    clang_getCompletionFixIt
```

Maybe your llvm version is too low(need >= 7.0.0).

- [ ] How to check llvm version ?

**_Ref:_** [Github Valloric/YouCompleteMe: When I enter VIM I get The ycmd server SHUT DOWN (restart with ':YcmRestartServer'). Unexpected error while loading the YCM core library. Type ':YcmToggleLogs ycmd_49739_stderr_rbhx3wqr.log' to check the logs #3236](https://github.com/Valloric/YouCompleteMe/issues/3236#issuecomment-439987788)

**_References:_**

- [博客园: 发现 vi 出现此错误~/.vim/bundle/YouCompleteMe/third_party/ycmd/ycm_core.so: undefined symbol: clang_getCompletionFixIt](https://www.cnblogs.com/dakewei/p/10491485.html)

### [Install] When build YouCompleteMe occur error `make[3]: *** No rule to make target '/home/rivergold/software/anaconda/lib/python3.7/config-3.7m-x86_64-linux-gnu/libpython3.7m.so', needed by '/home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycm_core.so'. Stop.`

When you build YouCompleteMe with Anaconda Python, maybe occur this error.

**_Ref:_** [CSDN: YouCompleteMe+anaconda+vim8.0 自动补全](https://blog.csdn.net/u013806541/article/details/72057272)

- [ ] The solution is not good.

<!--  -->
<br>

---

<!--  -->

## gruvbox

## Open Python file occur indent error in light red color

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-50747b88f91fa249.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="50%">
</p>

**Note:** It is not gruvbox error.

When set `syntax on`, python syntax will check invalid indent.

Add followings into `my_configs.vim` to close space error.

```vim
if exists('python_highlight_all')
    unlet python_highlight_all
endif
if exists('python_space_error_highlight')
    unlet python_space_error_highlight
endif
```

**_Ref:_** [stackoverflow: How to stop Vim highlighting trailing whitespace in python files](https://stackoverflow.com/a/47588983/4636081)

# Vim Shortcut

## Sacv and exit

- `ZZ`: Save and exit
- `ZQ`: Exit without saving

**_References:_**

- [How To Exit Vim? Multiple Ways To Quit Vim Editor](https://itsfoss.com/how-to-exit-vim/)

<!-- # SpaceVim

## Install

```bash
curl -sLf https://spacevim.org/install.sh | bash
```

***Ref:*** [SpaceVim doc: Quick start guide](https://spacevim.org/quick-start-guide/)

## Uninstall

```bash
curl -sLf https://spacevim.org/install.sh | bash -s -- --uninstall
```

***Ref:*** [Github SpaceVim/SpaceVim: 如何卸载 how to uninstall #941](https://github.com/SpaceVim/SpaceVim/issues/941#issuecomment-445451297)

## Config



Edit ``

```
relativenumber = false # Not use relative line number
``` -->
