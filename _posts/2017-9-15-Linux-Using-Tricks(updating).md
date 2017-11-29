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

## `cd` into previous path?<br>
```shell
cd -
```

## Using command change Ubuntu download source<br>
```shell
sed -is@http://archive.ubuntu.com/ubuntu/@https://mirrors.ustcedu.cn/ubuntu/@g /etc/apt/sources.list
```

## Install Java on Ubuntu with apt-get
```
sudo apt-add-repository ppa:webupd8team/java  
sudo apt-get update  
sudo apt-get install oracle-java8-installer  
```
***References:***
- [Blog: Installing Apache Spark on Ubuntu 16.04](https://www.santoshsrinivas.com/installing-apache-spark-on-ubuntu-16-04/)

## Differences of `ctr + d`, `ctr + z` and `ctr + c`<br>
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

## Inspect GPU state in unbuntu using `watch` ([\*ref](https://unix.stackexchange.com/questions/38560/gpu-usage-monitoring-cuda))
`watch -n 0.5 nvidia-smi`

***References:***
- [Unix & Linux: GPU usage monitoring (CUDA)](https://unix.stackexchange.com/questions/38560/gpu-usage-monitoring-cuda)

## Pass password to scp
Using `sshpass`
```
sudo apt-get install sshpass
```
```
sshpass -p <password> scp -P <port> <source path> <dist path>
```
- [stackoverflow: How to pass password to scp?](https://stackoverflow.com/questions/50096/how-to-pass-password-to-scp)

## Uninstall software
```
sudo apt-get purge <package name>
sudo apt-get autoremove
```
**Note:** It is dangerous to add `*` in `<package name>`, do not use `apt-get purge <package name*>`<br>
***References:***
- [ask ubuntu: What is the correct way to completely remove an application?](https://askubuntu.com/questions/187888/what-is-the-correct-way-to-completely-remove-an-application)

## Errors and Solutions
## Ubuntu error: 'apt-add-repository: command not found'<br>
```
apt-get install software-properties-common
```

## Ubuntu error: `Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)`<br>
```shell
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
```

## Ubuntu `.desktop` write format
Here is a example from `NetBeans`
```
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

<br>

* * *

<br>

# Common Software
## shadowsocks-qt-gui
```bash
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
2. Go to `System Settings` -> `Language Support`, change `Keyboard input method system as `fcitx`
3. Download Sogou input method from [here](https://pinyin.sogou.com/linux/?r=pinyin)
```
sudo dpkg -i install <sogou.deb>
```

## Gitkraken
Good git client on Linux os.
You can get the `.deb` from [here](https://www.gitkraken.com/) , and use `dpkg -i <gitkraken.deb>` to install it.


## Terminator
A powerful terminal for Linux.
```
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
    ```
    vim ~/.zshrc
    ```
    A recommending theme is `ys`, set `ZSH_THEME="ys"` in `.zshrc`.

***References***:
- [Github: robbyrussell/oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh)
- [LINUX大棚： Linux命令五分钟-用chsh选择shell](http://roclinux.cn/?p=739)

## Matlab
1. Download matlab2016b from [baidu cloud](https://pan.baidu.com/s/1mi0PRqK#list/path=%2F).
2. Mount install `.iso`
```
sudo mkdir /media/matlab
cd <matlab.iso path>
sudo mount -o loop <matlab-d1.iso> /media/matlab
```
3. Install
```
sudo /media/matlab/install
```
When it is in the middle of installation, `Eject DVD 1 and insert DVD 2 to continue.` will occur, you should mount `DVD2.iso`
```
sudo mount -o loop <matlab-d2.iso> /media/matlab
```

***Reference:***
- [ubuntu 16.04 安装 matlab](http://blog.csdn.net/jesse_mx/article/details/53956358)

## cmake-gui
- Ubuntu > 16.04
    ```
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
- [StackExchange-ask ubuntu: How to install cmake 3.2 on ubuntu 14.04?](https://askubuntu.com/questions/610291/how-to-install-cmake-3-2-on-ubuntu-14-04)

## System Monitor
This software is default installed on Ubuntu. It makes inspecting `disk/cpu/memory` use condition simply.
