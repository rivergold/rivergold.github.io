# Linux

## Linux查看硬盘分区

```bash
df -lh
```

***References:***

- [Linux 查看磁盘分区、文件系统、使用情况的命令和相关工具介绍](http://blog.51cto.com/13233/82677)

## Keep run wihout exist when using ssh to remote server

There are two solutions:

- Run with `nohup`
    ```bash
    nohup python file.py
    ```
- Run `tmux`, and using `tmux attach` to return to the same seesion
    ```bash
    tmux
    ```
    Then run what you want.

**Note:** Rivergold recommend the second method - using `tmux`

***References:***

- [stackExchange: How to keep a python script running when I close putty](https://unix.stackexchange.com/questions/362115/how-to-keep-a-python-script-running-when-i-close-putty)

## Count files num in a folder

- [Blog: ](https://blog.csdn.net/niguang09/article/details/6445778)

## Linux `tar`

- compress
    ```bash
    tar -czvf <file_name>.tar.gz <folder need compressed>
    tar -cjvf <file_name>.tar.bz2
    ```
- uncompress
    ```bash
    tar -xzvf <file_name>.tar.gz
    tar -xjvf <file_name>.tar.bz2
    ```

- `-c`: 建立一个压缩档案
- `-x`: 解压一个压缩档案
- `-z`: 是否具有`gzip`属性
- `-j`: 是否具有`bzip2`属性
- `-v`: 是否显示过程
- `-f`: 使用档名，需要参数的最后，后面立马接压缩包的名字

***References:***

- [Blog: tar压缩解压缩命令详解](https://www.cnblogs.com/jyaray/archive/2011/04/30/2033362.html)
- [Blog: linux tar (打包.压缩.解压缩)命令说明 | tar如何解压文件到指定的目录？](http://www.cnblogs.com/52linux/archive/2012/0luarocks install cutorch3/04/2379738.html)

## `Error mounting /dev/sdb1`

```bash
sudo ntfsfix /dev/sdb1 
```

***References:***

- [StackExchange: Error mounting /dev/sdb1 at /media/ on Ubuntu 14.04 LTS](https://askubuntu.com/questions/586308/error-mounting-dev-sdb1-at-media-on-ubuntu-14-04-lts)

## `ssh -X <user_name>@<ip>` occur error: `X11 forwarding request failed on channel 0`

1. `sudo yum install xorg-x11-xauth`
2. Change `/etc/ssh/sshd_config`
    ```bash
    X11Forwarding yes
    X11UseLocalhost no
    ```
3. Reload ssh config
    ```bash
    sudo service sshd restart
    ```
4. Install `cmake-gui` to have a try

***References:***

- [stackoverflow: X11 forwarding request failed on channel 0](https://stackoverflow.com/questions/38961495/x11-forwarding-request-failed-on-channel-0)
- [Ask Xmodulo: How to fix “X11 forwarding request failed on channel 0”](http://ask.xmodulo.com/fix-broken-x11-forwarding-ssh.html)
- [StackExchange: ssh returns message “X11 forwarding request failed on channel 1”](https://unix.stackexchange.com/questions/111519/ssh-returns-message-x11-forwarding-request-failed-on-channel-1)

## Problems & Solutions

### `rm -rf *` recovery

- [blog: CentOS 恢复 rm -rf * 误删数据](https://my.oschina.net/coda/blog/222670)

## CentOS

### `rpm`的使用

- [Blog: CentOS RPM 包使用详解](http://blog.51cto.com/tengq/1930181)

### vim version update

vim 7 not support `YouCompleteMe`, need to update to version 8

***References:***

- [TecMint: Vim 8.0 Is Released After 10 Years – Install on Linux Systems](https://www.tecmint.com/vim-8-0-install-in-ubuntu-linux-systems/)

<!--  -->

<br>

***
<!--  -->

# Tools

## Using `pip` install **OpenCV**

- [Medium: Installing OpenCV 3.3.0 on Ubuntu 16.04 LTS](https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961)


<!--  -->
<br>

***
<!--  -->


# Frameworks

## Torch

### Install

- [torch: Getting started with Torch](http://torch.ch/docs/getting-started.html)

When CentOS install torch, it occur error: **xxxx**

when install torch error: https://ubuntuforums.org/showthread.php?t=1670531

### Tensor slice index

```lua
M:sub(1, -1, 2, 3)
```

***References:***

- [Blog: torch Tensor学习：切片操作](https://www.cnblogs.com/YiXiaoZhou/p/6387769.html)

### Torch print dimension size of each layer

***References:***

- [Github torch/nn/Issues: How to print the output dimension size of each layer, just like "Top shape" in caffe](https://github.com/torch/nn/issues/922)

### Torch read weights of the model

```lua
model.modules[2].weight
```

***References:***

- [stackoverflow: [torch]how to read weights in nn model](https://stackoverflow.com/questions/32086106/torchhow-to-read-weights-in-nn-model)

### Problems and Solutions

- Install [torch-opencv](https://github.com/VisionLabs/torch-opencv) ouucurs errors: `Cannot Find OpenCV` or other errors about OpenCV
    **Solution:** Change `CMakeLists.txt` and set OpenCV path. Then use `luarocks make` to build and install package.
    ***References:*** 
    - [isionLabs/torch-opencv: Problem installing from source](https://github.com/VisionLabs/torch-opencv/issues/182)

## Lua

> Lua arrays begin with 1

### Getting input from the user in Lua

- [stackoverflow: Getting input from the user in Lua](https://stackoverflow.com/questions/12069109/getting-input-from-the-user-in-lua) 

## Deep Learning Model Convertor

- [Github: ysh329/deep-learning-model-convertor](https://github.com/ysh329/deep-learning-model-convertor)

# PyTorch

## PyTorch load torch model

```python
import torch
from torch.utils.serialization import load_lua
model = load_lua(<torch model>)
```

***References:***

- [PyTorch Forum: Convert/import Torch model to PyTorch](https://discuss.pytorch.org/t/convert-import-torch-model-to-pytorch/37)


<!--  -->

<br>

***
<!--  -->

# Python

## Basics

### `zip`

- Combine two list:
    ```python
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = list(zip(a, b))
    print(c)
    >>> [(1, 4), (2, 5), (3, 6)]
    ```
    **理解:** `zip`就是将多个list对应位置上的element分别组合起来
- Unzip a list of list/tuple
    ```python
    a = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    a1, a2, a3 = zip(*a)
    print(a1, a3, a3)
    >>> (1, 4, 7) (2, 5, 8), (3, 6, 9)
    ```

    **Note:** If you regard `a` as a matrix, `zip(*a)` will get each column of `a`.

***References:***

- [stackoverflow: How to unzip a list of tuples into individual lists? [duplicate]](https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists/12974504)
- [stackoverflow: What does the Star operator mean? [duplicate]](https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean)

### collections

#### `defaultdict`

- [简书：Python中collections.defaultdict()使用](https://www.jianshu.com/p/26df28b3bfc8)

### Issues

- [Global, Local and nonlocal Variables](https://www.python-course.eu/python3_global_vs_local_variables.php)

## Numpy

### Copy in numpy

***References:***

- [CSDN:【Python】numpy中的copy问题详解](https://blog.csdn.net/u010099080/article/details/59111207)

### `np.newaxis` add new axis to array

```python
x = np.array([1, 2, 3, 4])
x = x[:, np.newaxis]
print(x)
>>> array([[1],
           [2],
           [3],
           [4],
           [5]])
x = np.array(range(4)).reshape(2, 2)
x = x[..., np.newaxis]
print(x.shape)
>>> (2, 2, 1)
```

***References:***

- [stackoverflow: How does numpy.newaxis work and when to use it?](https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it)

### round number

***References:***
- [SciPy.org: numpy.around](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.around.html)

### Print full numpy array

```python
import numpy as np
np.set_printoptions(threshold=np.nan)
```

***References:***

- [stackoverflow: How to print the full NumPy array?](https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array)

### Change set a value for a region of array

```python
a = np.zeros((2, 2))
b = np.ones((2, 2))
b[0:2, 0:2] = a
```

***References:***

- [stackoverflow: Numpy array and change value regions](https://stackoverflow.com/questions/13000842/numpy-array-and-change-value-regions)

## Pandas

### Create `pd.DataFrame` with setting column names

```python
df = pd.DataFrame(<data>, columns=[name_1, name_2, ..., name_n])
```

### Rename DataFrame columns name

Using `df.rename(index=str, columns={<pre_name_1>: <new_name_1>, <pre_name_2>: <new_name_2>}`

```python
# Rename
df = df.rename(index=str, columns={'mask_coordinates': 'mask', 'resolution_processed': 'resolution'})
```

***References:***

- [pandas doc: pandas.DataFrame.rename](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html)

### `SettingWithCopyWarning`

- [SofaSOfa.io: pandas DataFrame中经常出现SettingWithCopyWarning](http://sofasofa.io/forum_main_post.php?postid=1001449)

### df.loc[]

If you want to using `df.loc[]` by index and column name, you should set a column named `index` of df

```python
df['index'] = range(len(df))
df.set_index('index')
```

Then you can use like this:

```python
df.loc[:10, '<col_name>']
```

**Note:** `df.loc[:10]` will contain the index=10, the 11th element, it is different between Python.

## OpenCV

### When OpenCV read color image, color information is stored as BGR, in order to convert to RGB

```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(<img_path>) # BRG
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```

***References:***

- [PHYSIOPHILE: WHY OPENCV USES BGR (NOT RGB)](https://physiophile.wordpress.com/2017/01/12/why-opencv-uses-bgr-not-rgb/)

### When using OpenCV in IPython or Jupyter notebook, `cv2.imshow` and `cv2.waitKey(0)` cause crash.

Solution: Add `cv2.destroyAllWindows()`

```python
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

***References:***

- [Github Opencv/opencv Issues: opencv cv2.waitKey() do not work well with python idle or ipython](https://github.com/opencv/opencv/issues/6452)

## Anaconda

### When install Anaconda/Miniconda, occur error `bunzip2: command not found

You need to install `bzip2`

```bash
sudo apt-get install bzip2
```

<!--  -->
<br>

***
<!--  -->


# Vim

## Vim packages manager [Vundle](https://github.com/VundleVim/Vundle.vim)

***References:***

- [简书：使用Vim插件管理器Vundle](https://www.jianshu.com/p/8d416ac4ad11)

## `.vimrc` configurion

```vim
set paste " 取消粘贴时自动注释
set nu!   " 显示行号
set ts=4  " 设置tab键为4个空格
set backspace=2 " 设置backspace删除
```

## Install vim 8 or Update vim 7 to vim 8

Best way to install vim 8 on linux is building vim from [source code](https://github.com/vim/vim). And you should build vim with `python` and the `python` must be the original one, **not Anaconda**. 

```bash
sudo yum install -y ruby ruby-devel lua lua-devel luajit \
luajit-devel ctags git python python-devel \
python3 python3-devel tcl-devel \
perl perl-devel perl-ExtUtils-ParseXS \
perl-ExtUtils-XSpp perl-ExtUtils-CBuilder \
perl-ExtUtils-Embed
```

```bash
./configure --with-features=huge \
            --enable-multibyte \
            --enable-rubyinterp=yes \
            --enable-pythoninterp=yes \
            --with-python-config-dir=/usr/lib/python2.7/config \
            --enable-perlinterp=yes \
            --enable-luainterp=yes \
            --enable-gui=gtk2 \
            --enable-cscope \
            --prefix=/usr/local
```

***References:***

- [Github Valloric/YouCompleteMe Wiki: Building Vim from source](https://github.com/Valloric/YouCompleteMe/wiki/Building-Vim-from-source)
- [Blog: YouCompleteMe+anaconda+vim8.0自动补全](https://blog.csdn.net/u013806541/article/details/72057272)

### Install `YouCompleteMe` from source

- [YouCompleteMe](https://valloric.github.io/YouCompleteMe/#ubuntu-linux-x64)

## Problems & Solutions

### Error `swap file <xxxx.swap> already exists`

Remove the `<.swap>` file.

### Error when using vim to edit `.py`: `TabError: Inconsistent use of tabs and spaces in indentation`

Set vim to convert tab into spaces

```vim
: set expandtab
```

***References:***

- [stackoverflow: Replace Tab with Spaces in Vim](https://stackoverflow.com/questions/426963/replace-tab-with-spaces-in-vim)

### Vim `Backspace` cannot delete

```vim
set backspace=2 " make backspace work like most other programs
```

***References:***

- [FANDOM Vim Tips Wiki: Backspace and delete problems](http://vim.wikia.com/wiki/Backspace_and_delete_problems)

### vundle install `YouCompleteMe`

Report `"YouCompleteMe unavailable: requires Vim compiled with Python 2.x support" error`

***References:***

- [Github Valloric/YouCompleteMe Issue Report "YouCompleteMe unavailable: requires Vim compiled with Python 2.x support" error #1907](https://github.com/Valloric/YouCompleteMe/issues/1907)
- [Blog: CENTOS7安装VIM插件YOUCOMPLETEME](http://dreamlikes.cn/archives/940)

<!--  -->
<br>

***
<!--  -->


# Other Tricks

## How to Generate ssh-key

```bash
ssh-keygen -t rsa -C "your_email@example.com"
```

- [Bitbucket Support: Creating SSH keys](https://confluence.atlassian.com/bitbucketserver/creating-ssh-keys-776639788.html)
- [Github Help: Connecting to GitHub with SSH](https://help.github.com/articles/connecting-to-github-with-ssh/)

## Using command install miniconda silent

```bash
bash <miniconda.sh path> -b -p <install path>
```

- [Conda doc: Installing on macOS](https://conda.io/docs/user-guide/install/macos.html#install-macos-silent)

## `pip` install OpenCV packages for Python

```bash
pip install opencv-python # If you need only main modules
pip install opencv-contrib-python # If you need both main and contrib modules
```

***References:***

- [PyPi: opencv-python 3.4.1.15](https://pypi.org/project/opencv-python/)

## Nvidia cuDNN download websites

- [Nvidia: cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

## Linux build OpenCV from source, occur cannot download `ippicv`

**Solution:** Manually download `ippicv` from [here](https://raw.githubusercontent.com/Itseez/opencv_3rdparty/81a676001ca8075ada498583e4166079e5744668/ippicv/ippicv_linux_20151201.tgz)
And move it into `opencv-3.1.0/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e`

***References:***

- [Blog: Cenots7编译Opencv3.1错误：下载ippicv，解决方案](https://blog.csdn.net/daunxx/article/details/50495111)

## Cuda version manage

- [stackoverflow: How to change CUDA version](https://stackoverflow.com/questions/45477133/how-to-change-cuda-version)
- [Blog: 安装多版本 cuda ，多版本之间切换](https://blog.csdn.net/Maple2014/article/details/78574275)

## Terminator

- [Blog: Ubuntu终端多窗口分屏Terminator](https://blog.csdn.net/MrGong_/article/details/77817018)
- [stackoverflow: How do I set default terminal to terminator? [closed]](https://stackoverflow.com/questions/16808231/how-do-i-set-default-terminal-to-terminator)

### Set `Open in Terminator` in right click

- [stackoverflow: Setting nautilus-open-terminal to launch Terminator rather than gnome-terminal](https://askubuntu.com/questions/76712/setting-nautilus-open-terminal-to-launch-terminator-rather-than-gnome-terminal)

### Config

- [Wentong's Blog: 使用Terminator增强你的终端](http://blog.wentong.me/2014/05/work-with-terminator/)

Change font size

```config
use_system_font = False
font = Monospace 12
```

## `tmux`

- [Blog: 优雅地使用命令行：Tmux 终端复用](https://harttle.land/2015/11/06/tmux-startup.html)

<!--  -->
<br>

***
<!--  -->

# VSCode
VSCode Setting Sync
- [如何使用 VSCode 的 Setting Sync 插件](https://segmentfault.com/a/1190000010648319)

## shortcut

- `ctr` + `k`, then `o`: Open current file in new window
- `ctr` + \`:  Open terminal

***Referneces:***
- [VS Code折腾记 - (2) 快捷键大全，没有更全](https://blog.csdn.net/crper/article/details/54099319)

# 临时

Tensorflow: 实战Google深度学习框架PDF

- [网址](http://www.olecn.com/5096.html)
- [百度网盘地址](http://www.olecn.com/download.php?id=5096)

Hands-On Machine Learning with Scikit-Learn & Tensorflow

- [Github apachecn/hands_on_Ml_with_Sklearn_and_TF](https://github.com/apachecn/hands_on_Ml_with_Sklearn_and_TF)


## Windows远程登录Linux并显示图像

- [Blog: windows 远程登陆linux并显示图像界面](https://blog.csdn.net/u013203733/article/details/65444084)

## Understanding Region-based Fully Convolutional Networks(R-FCN) for object detection

- [Medium Blog: Understanding Region-based Fully Convolutional Networks(R-FCN) for object detection](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99)

## Ubuntu install Eclipse

- [Blog: Ubuntu 16.04安装Eclipse + C/C++开发环境配置](https://blog.csdn.net/colin_lisicong/article/details/70939143)

# VNC

- VNC client
- VNC server

## What is VNC

***References:***

- [StackExchange: Differences between VNC and ssh -X](https://unix.stackexchange.com/questions/1960/differences-between-vnc-and-ssh-x)

## Install VNC-server on remote server

1. Install `GNOME Desktop`
2. Install `tigervnc-server`
3. Config
    `~/.vnc/xstartup`
    ```bash
    #!/bin/sh
    [ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
    [ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
    xsetroot -solid grey
    vncconfig -iconic &
    x-terminal-emulator -geometry 80x24+10+10 -ls -title "$VNCDESKTOP Desktop" &
    x-window-manager &
    ```

***References:***

- [HowtoForge: VNC-Server installation on CentOS 7](https://www.howtoforge.com/vnc-server-installation-on-centos-7)
- [CentOS: VNC(虚拟网络计算)](https://wiki.centos.org/zh/HowTos/VNC-Server)
- [StackExchange: Why VNC not showing actual Remote Desktop](https://unix.stackexchange.com/questions/61750/why-vnc-not-showing-actual-remote-desktop)
- [NDC HOST: How To Install X Server on a VPS (with VNC access)](https://www.ndchost.com/wiki/vps/x-server-vnc)
- [怎样在 CentOS 7.0 上安装和配置 VNC 服务器](https://linux.cn/article-5335-1.html)
- [CentOS7.2安装VNC，让Windows远程连接CentOS 7.2 图形化界面](http://blog.51cto.com/12217917/2060252)
