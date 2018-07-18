This article has recorded some tips and tricks for linux, especially for Ubuntu. All of these commands and tricks are sorted up with my using experience.

# Linux Command Tips

## Update `.barchrc`

```shell
. ~/.bashrc
```

Another way

```
source ~/.bashrc
```

***References:***

- [stackoverflow: How do I reload .bashrc without logging out and back in?](https://stackoverflow.com/questions/2518127/how-do-i-reload-bashrc-without-logging-out-and-back-in)

## Rename a directory

```shell
mv <oldname> <newname>
```

***References:***

- [ask ubuntu: How do I rename a directory via the command line?](https://askubuntu.com/questions/56326/how-do-i-rename-a-directory-via-the-command-line)

## `cd` into previous path

```shell
cd -
```

## Using command change Ubuntu download source<br>

```shell
sed -is@http://archive.ubuntu.com/ubuntu/@https://mirrors.ustcedu.cn/ubuntu/@g /etc/apt/sources.list
sed -i s@http://security.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g /etc/apt/sources.list
```

## Install Java on Ubuntu with apt-get

```
sudo apt-add-repository ppa:webupd8team/java  
sudo apt-get update  
sudo apt-get install oracle-java8-installer  
```

***References:***

- [Blog: Installing Apache Spark on Ubuntu 16.04](https://www.santoshsrinivas.com/installing-apache-spark-on-ubuntu-16-04/)

## Differences of `ctr + d`, `ctr + z` and `ctr + c`

- `ctr + d`: terminate input or exit the terminal or shell
- `ctr + z`: suspend foreground processes
- `ctr + c`: kill foregrousortednd processes

## Change `pip` and `conda` donload source

- `pip`
    1. Create a folder named `.pip` in `~/`
    2. Create a file named `pip.conf`
    3. Write the followings into `pip.conf`
    ```bash
    [global]
    index-url=https://pypi.douban.com/simple
    ```

- `conda`
    Input followings in terminal
    ```shell
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --set show_channel_urls yes
    ```
    And it will create a file named `.condarc` in `~/` folder.

## `tar`

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

## Send/Copy files between *Windows* and *Linux*

1. Install SSH Clients on windows client computer
2. Using `ssh` can visit and log remote linux host
3. `pscp` command can send/download files to/from remote linux host
    ```shell
    pscp [options] source de
    ```

- send file to linux host
    ```shell
    pscp <windows file path> <linux user-name>@<ip>:<path>
    ```

- copy file from linux host
    ```shell
    pscp <linux user-name>@<ip>:<file path> <windows path>
    ```

## `wget` command

```shell
wget [options] <url>
```

| Options |  whole  |                Description                         |
|:--:|:------------:|:--------------------------------------------------:|
| -c | --continue | Continue getting a partially-downloaded file. |


***References:***

- [每天一个linux命令（61）：wget命令](http://www.cnblogs.com/peida/archive/2013/03/18/2965369.html)
- [Computer Hope: Linux wget command](https://www.computerhope.com/unix/wget.htm)

## Inspect GPU state in unbuntu using `watch`

`watch -n 0.5 nvidia-smi`

***References:***

- [Unix & Linux: GPU usage monitoring (CUDA)](https://unix.stackexchange.com/questions/38560/gpu-usage-monitoring-cuda)

## Pass password to scp

Using `sshpass`

```bash
sudo apt-get install sshpass
```

```bash
sshpass -p <password> scp -P <port> <source path> <dist path>
```

- [stackoverflow: How to pass password to scp?](https://stackoverflow.com/questions/50096/how-to-pass-password-to-scp)

## Uninstall software

```bash
sudo apt-get purge <package name>
sudo apt-get autoremove
```

**Note:** It is dangerous to add `*` in `<package name>`, do not use `apt-get purge <package name*>`<br>

***References:***

- [ask ubuntu: What is the correct way to completely remove an application?](https://askubuntu.com/questions/187888/what-is-the-correct-way-to-completely-remove-an-application)

## Create a new user

1. 
    If you are a `root` user,
    ```sh
    adduser <username> 
    ```
    If you are a `non-root` user,
    ```bash
    sudo adduser <username>
    ``` 
2. Set a passward 
    ```bash
    Set password prompts:
    Enter new UNIX password:
    Retype new UNIX password:
    passwd: password updated successfully
    ```
3. Follow the prompts to set the new user's information. It is fine to accept the defaults to leave all of this information blank.
4. Use `usermod` command to add the user into the `sudo` group, otherwise the user cannot use `sudo`

```bash
usermod -aG sudo <username>
```

***References:***

- [DigtialOcean: How To Create a Sudo User on Ubuntu [Quickstart]](https://www.digitalocean.com/community/tutorials/how-to-create-a-sudo-user-on-ubuntu-quickstart)

## Ubuntu `.desktop` write format

Here is a example from `NetBeans`

```bash
[Desktop Entry]
Encoding=UTF-8
Name=NetBeans IDE 8.2
Comment=The Smarter Way to Code
Exec=/bin/sh "/home/rivergold/software/NetBeans/netbeans-8.2/bin/netbeans"
Icon=/home/rivergold/software/NetBeans/netbeans-8.2/nb/netbeans.png
Categories=Application;Development;Java;IDE
Version=1.0
Type=Application
Terminal=0
```

And then, run `chmod +x <name>.desktop`

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

<!--  -->
<br>

***
<!--  -->

# Errors and Solutions

## Ubuntu error: 'apt-add-repository: command not found'<br>

```shell
apt-get install software-properties-common
```

## Ubuntu error: `Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)`<br>

```shell
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
```

## `System program problem detected` occur when Ubuntu starts.

```
sudo rm /var/crash/*
```

***References:***

- [ask ubuntu: Getting “System program problem detected” pops up regularly after upgrade](https://askubuntu.com/questions/133385/getting-system-program-problem-detected-pops-up-regularly-after-upgrade/369297)

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

<!--  -->
<br>

***
<!--  -->

# Common Software

## shadowsocks-qt-gui

```shell
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo apt-get update
sudo apt-get install shadowsocks-qt5
```

And you also need to install Chrome or Firefox extension `switchyomega`, you can get it from [github: FelisCatus/SwitchyOmega](https://github.com/FelisCatus/SwitchyOmega/releases). And the configuration can be get from its [wiki](https://github.com/FelisCatus/SwitchyOmega/wiki/GFWList).

## synergy

```bash
sudo apt-get install quicksynergy
```

## Filezilla

```bash
sudo apt-get install filezilla
```

## sogou input

1. First install `fcitx`. Because Ubuntu set `Ibus` as default keyboard input framework, we need change it.
    ```bash
    sudo apt-get install fcitx
    ```
2. Go to `System Settings` -> `Language Support`, change `Keyboard input method system` as `fcitx`
3. Download Sogou input method from [here](https://pinyin.sogou.com/linux/?r=pinyin)
    ```shell
    sudo dpkg -i install <sogou.deb>
    ```

## Gitkraken

Good git client on Linux os.
You can get the `.deb` from [here](https://www.gitkraken.com/) , and use `dpkg -i <gitkraken.deb>` to install it.

**Problem&Solution:** If you double click the software icon, there is nothing happened. You'd better start the software by using command in terminal, and find what wrong during starting.

- Ubuntu 18.04 can not start `Gitkraken`. Occur error about `libgnome-keyring.so.0: cannot open shared object file: No such file or directory`

    ***References:***

    - [StackExchange: Error running Gitkraken even though dependencies installed](https://superuser.com/questions/1233459/error-running-gitkraken-even-though-dependencies-installed)

## Terminator

A powerful terminal for Linux.

```shell
sudo apt-get install terminator
```

Configuration tips:

- Click right button and choose `Preference` to set base configuration.
- `vim ~/.config/terminator/config` to set custom configuration, such as change window size:

    ```bash
      [layouts]
      [[default]]
        [[[child1]]]
          parent = window0
          type = Terminal
        [[[window0]]]
          parent = ""
          # Add this line to set window size
          size = 800, 500
    ```
- Set `Terminator` can be opened by right clicked by using `nautilus-actions`
  First install `nautilus-actions`
    ```bash
    sudo apt-get install nautilus-actions -y
    ```
    **Note:** In Ubuntu 18.04, `nautilus-actions` is named as `filemanager-actions`, you should install by:
    ```shell
    sudo add-apt-repository ppa:daniel-marynicz/filemanager-actions
    sudo apt-get install filemanager-actions-nautilus-extension # Nautilus
    ```
    And set like this,
    <p align="center">
      <img src="http://ovvybawkj.bkt.clouddn.com/linux/nautilus-actions-1" width="80%">
    </p>
    Set Terminator path and parameters `--working-directory=%d/%b`
    <p align="center">
      <img src="http://ovvybawkj.bkt.clouddn.com/linux/nautilus-actions-2" width="80%">
    </p>

**Some shortcuts**

- Split terminals horizontally: `Ctrl + Shift + o`
- Split terminals vertically: `Ctrl + Shift + E`
- Move to the terminal above the current one: `Alt + ↑`
- Move to the terminal below the current one: `Alt + ↓`
- Move to the terminal left the current one: `Alt + ←`
- Move to the terminal left the current one: `Alt + →`
- Open a new terminal in current window: `Ctrl + Shit + T`
- Net tab terminal: `Ctrl + PageUp` or `Ctrl + PageDown`

***References***

- [简书： 5分钟入手Terminator](http://www.jianshu.com/p/cee2de32ca28)
- [ubuntu: Setting nautilus-open-terminal to launch Terminator rather than gnome-terminal](https://askubuntu.com/questions/76712/setting-nautilus-open-terminal-to-launch-terminator-rather-than-gnome-terminal)
- [stackoverflow: Nautilus-actions in 18.04](https://askubuntu.com/questions/1030940/nautilus-actions-in-18-04)

## zsh and oh-my-zsh

`zsh` is powerfull and beautiful shell for Unix, and `Oh my zsh` is an open source, community-driven framework for managing `zsh` configuration.

1. Install and set up `zsh` as default
    ```bash
    sudo apt-get isntall zsh
    # Set zsh as default shell
    chsh -s $(which zsh)
    ```
2. Install `oh my zsh`
    ```bash
    sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
    ```
3. Config
    ```shell
    vim ~/.zshrc
    ```
    A recommending theme is `ys`, set `ZSH_THEME="ys"` in `.zshrc`.

***References***:

- [Github: robbyrussell/oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh)
- [LINUX大棚： Linux命令五分钟-用chsh选择shell](http://roclinux.cn/?p=739)

## Matlab

1. Download matlab2016b from [baidu cloud](https://pan.baidu.com/s/1mi0PRqK#list/path=%2F).
2. Mount install `.iso`
    ```shell
    sudo mkdir /media/matlab
    cd <matlab.iso path>
    sudo mount -o loop <matlab-d1.iso> /media/matlab
    ```
3. Install
    ```shell
    sudo /media/matlab/install
    ```
    When it is in the middle of installation, `Eject DVD 1 and insert DVD 2 to continue.` will occur, you should mount `DVD2.iso`
    ```shell
    sudo mount -o loop <matlab-d2.iso> /media/matlab
    ```

***Reference:***

- [ubuntu 16.04 安装 matlab](http://blog.csdn.net/jesse_mx/article/details/53956358)

## cmake-gui

- Ubuntu > 16.04
    ```shell
    sudo apt-get install cmake-qt-gui
    ```
- Ubuntu 14.04
    Default cmake version of 14.04 is 2.8 which is too old.
    ```bash
    sudo apt-get install software-properties-common
    sudo add-apt-repository ppa:george-edison55/cmake-3.x
    sudo apt-get update
    sudo apt-get install cmake-qt-gui
    ```

***References:***

- [StackExchange-ask ubuntu: How to install cmake 3.2 on ubuntu 14.04?](https://askubuntu.com/q./configure --with-features=huge \
--enable-multibyte \
--enable-rubyinterp=yes \
--enable-python3interp=yes \
--with-python3-config-dir=~/software/anaconda/bin/ \
--enable-perlinterp=yes \
--enable-luainterp=yes \
--enable-gui=gtk2 \
--enable-cscope \
--prefix=/usr/localuestions/610291/how-to-install-cmake-3-2-on-ubuntu-14-04)

## System Monitor

This software is default installed on Ubuntu. It makes inspecting `disk/cpu/memory` use condition simply.

## Netease cloud music(网易云音乐)

When start netease-cloud-music occur error `Local file: "" ("netease-cloud-music")`, you must start netease cloud music with `sudo`. One way to solve this trouble is to set `alias music="sudo netease-cloud-music"`

***References:***

- [知乎: Ubuntu 18.04 网易云音乐无法打开问题解决方案](https://zhuanlan.zhihu.com/p/37324458)

- [知乎: 在Ubuntu 上有什么必装的实用软件？](https://www.zhihu.com/question/19811112)

## Stardict

Powerful dictionary on Linux.

***References:***

- [StarDict Dictionaries -- 星际译王词库 词典下载](http://download.huzheng.org/)
- [Ubuntu 14.04 安装配置强大的星际译王（stardict）词典](https://blog.csdn.net/tecn14/article/details/25917149)

## tmux

> tmux is a terminal multiplexer: it enables a number of terminals to be created, accessed, and controlled from a single screen. tmux may be detached from a screen and continue running in the background, then later reattached.

One useful function of `tmux` is **Keep run wihout exist when using ssh to remote server**

Ubuntu can install by `sudo apt-get install tmux`

**tmux tips:**

- All tmux shortcuts need hit prefix `ctrl + b` and then
- `d`: detach (tmux still run)

***References:***

- [GithubL ryerh/tmux-cheatsheet.markdown](https://gist.github.com/ryerh/14b7c24dfd623ef8edc7)
- [Github: henrik/tmux_cheatsheet.markdown](https://gist.github.com/henrik/1967800)

## vim

### Install

You'd better install Vim with `python` support. You can use `vim --version` to check if vim is installed with `python`

Build `vim` from [source](https://github.com/vim/vim/releases):

1. Install `python-dev`
    - Ubuntu
        ```shell
        sudo apt-get install python3-dev
        ```
    - Centos
        ```shell
        yum install python36-devel
        ```
    ***Referneces:***
    - [ubuntu packages: python3-dev (3.5.1-3)](https://packages.ubuntu.com/xenial/python3-dev)
    - [stackoverflow: How to install python3-devel on red hat 7](https://stackoverflow.com/questions/43047284/how-to-install-python3-devel-on-red-hat-7)


2. `cd` into vim folder
3. Config
    ```shell
    ./configure --with-features=huge \
    --enable-multibyte \
    --enable-rubyinterp=yes \
    --enable-python3interp=yes \
    --with-python3-config-dir=~/software/anaconda/bin/ \
    --enable-perlinterp=yes \
    --enable-luainterp=yes \
    --enable-gui=gtk2 \
    --enable-cscope \
    --prefix=/usr/local
    ```
    **Note:** vim can be built with python2 and python3, but when install `Youcompleteme`, it need only one python version. So, when you build vim, you'd better only choose on python version. If you want to build vim with python2, you need to change `--enable-python3interp=yes` to `--enable-pythoninterp=yes` and `--with-python3-config-dir=~/software/anaconda/bin/` to `--with-python-config-dir=/usr/lib/python2.7/config`
4. `make -j8` and `sudo make install`
5. Config `~/.vimrc`
    ```vim
    set paste " 取消粘贴时自动注释
    set nu! " 显示行号
    set expandtab "tab使用空格替换
    set ts=4 " 设置tab键为4个空格
    syntax on " 代码高亮
    set backspace=2

    set nocompatible              " be iMproved, required
    filetype off                  " required
**Config Youcompleteme**
    " set the runtime path to include Vundle and initialize
    set rtp+=~/.vim/bundle/Vundle.vim
    call vundle#begin()
    " alternatively, pass a path where Vundle should install plugins
    "call vundle#begin('~/some/path/here')

    " let Vundle manage Vundle, required
    Plugin 'VundleVim/Vundle.vim'

    Plugin 'Valloric/YouCompleteMe'
    "Plugin 'davidhalter/jedi-vim'

    " All of your Plugins must be added before the following line
    call vundle#end()            " required
    filetype plugin indent on    " required
    " To ignore plugin indent changes, instead use:
    "filetype plugin on
    "
    " Brief help
    " :PluginList       - lists configured plugins
    " :PluginInstall    - installs plugins; append `!` to update or justluginUpdate
    " :PluginSearch foo - searches for foo; append `!` to refresh local cache
    " :PluginClean      - confirms removal of unused plugins; append `!` toto-approve removal
    "
    " see :h vundle for more details or wiki for FAQ
    " Put your non-Plugin stuff after this line
    ```


### Problems & Solutions

- Centos 7.2 install `python36-devel` occur error `Requires: libcrypto.so.10(openssl.1.0.2)(64bit)`
    You need to update openssl from `1.0.1` to `1.0.2`. You nend download [openssl-libs-1.0.2k-12.el7.x86_64.rpm](https://centos.pkgs.org/7/centos-x86_64/openssl-libs-1.0.2k-12.el7.x86_64.rpm.html) and run `rpm -i openssl-libs-1.0.2k-12.el7.x86_64.rpm`. And then it may occur another error like `conflicts with file from package`, you can use `rpm -i --replacefiles openssl-libs-1.0.2k-12.el7.x86_64.rpm`

    ***References:***
    - [Blog: 解决CentOS下的conflicts with file from错误.](http://rayfuxk.iteye.com/blog/2280643)

### Plugins

#### Vundle

Install

```shell
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```

#### Youcompleteme

You can install `Youcompleteme` using `vundle`, but it will take a long time. Here, we install it from source.

1. **Make sure** your vim is build with python2 or python3
2. Clone `Youcompleteme` source code into `~/.vim/bundle/`**Config Youcompleteme**
    ```shell
    git clone --recursive https://github.com/Valloric/YouCompleteMe.git
    ```
3. `cd ~/.vim/bundle/YouCompleteMe`
    ```shell
    python ./install.py
    ```**Config Youcompleteme**
    **N**Config Youcompleteme** here you need to use python2, if you**Config Youcompleteme**d to use python3

***Refe**Config Youcompleteme**

- [Gith**Config Youcompleteme**(https://github.com/Valloric/YouCompleteMe#ubuntu-linux-x64)
- [简书: 一步一步带你安装史上最难安装的 vim 插件 —— YouCompleteMe](https://www.jianshu.com/p/d908ce81017a)

**Problems & Solutions**

- Using python3 run `python ./install.py` occur error `AttributeError: module 'enum' has no attribute 'IntFlag'`
    You need to uninstall `enum34` by run `pip uninstall enum34`
    ***References:***
    - [stackoverflow: Why Python 3.6.1 throws AttributeError: module 'enum' has no attribute 'IntFlag'?](https://stackoverflow.com/questions/43124775/why-python-3-6-1-throws-attributeerror-module-enum-has-no-attribute-intflag)

- Using python3 run `python ./install.py` occur error `File "/root/software/anaconda/lib/python3.6/site-packages/uuid.py", line 138 if not 0 <= time_low < 1<<32L:`
    I solve this error by `pip uninstall uuid`

### Configuration

#### Basic

```vim
set paste " 取消粘贴时自动注释
set nu! " 显示行号
set expandtab "tab使用空格替换
set ts=4 " 设置tab键为4个空格
syntax on " 代码高亮
set backspace=2 " 设置backspace可以删除
```

#### Youcompleteme

```vim
" Disable Preview Window
set completeopt-=preview
let g:ycm_add_preview_to_completeopt = 0
```

***References:***

- [Blog: ubuntu下安装自动补全YouCompleteMe](https://www.cnblogs.com/litifeng/p/6671446.html)
- [Github: Valloric/YouCompleteMe Issue: Auto close preview window after insertion not closing sometimes](https://github.com/Valloric/YouCompleteMe/issues/524)