This article has recorded some tips and tricks for linux, especially for Ubuntu. All of these commands and tricks are sorted up with my using experience.

# :fallen_leaf:Basics

## Partition

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-05de2dcd0ea5ad0f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="100%">
</p>

Ref [鸟哥的 Linux 私房菜]()

### Install Ubuntu with UEFI

If the other OS installed on your computer is UEFI, you must install Ubuntu with UEFI, if the OS installed with BIOS, you must install Ubuntu with BIOS.

**_References:_**

- [Ubuntu doc: UEFI](https://help.ubuntu.com/community/UEFI#Set_up_the_firmware_in_UEFI_or_BIOS.2FCSM.2FLegacy_mode)

<!--  -->
<br>

---

<!--  -->

## shell Pipe

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-f2ee87b539717d93.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="50%">
</p>

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-acfdf915e317f6bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="90%">
</p>

**Example:**

```shell
echo a | sudo -S apt search xxx
```

**_References:_**

- [Linux 学习: 聊聊 Linux 的匿名管道](https://mp.weixin.qq.com/s?__biz=MzI4MDEwNzAzNg==&mid=2649445728&idx=2&sn=72f798eb75f4d6ae096a248f5f81be3d&chksm=f3a27413c4d5fd0549b1dc7e3aa81b06f25ba6e9b3149ba66412dc26ab940bbdf11750ab6026&mpshare=1&scene=1&srcid=#rd)

- [linux shell 管道命令(pipe)使用及与 shell 重定向区别](https://www.cnblogs.com/chengmo/archive/2010/10/21/1856577.html)

## :triangular_flag_on_post:shell Redirect

**_References:_**

- [CSDN: Linux 中>、>>](https://blog.csdn.net/hellozpc/article/details/46721811)

- [Linux 中国: 理解 Bash 中的尖括号](https://zhuanlan.zhihu.com/p/56210435)

- 0: stdin 标准输入
- 1: stdout 标准输出
- 2: stderr 标准错误

:thumbsup: `ls 2>&1 > f.out`: 将 stderr 重定向到标准输出，再重定向到文件中

**_Ref:_** [CSDN: make 2>&1 | tee log.txt 命令解析](https://blog.csdn.net/Dr_Unknown/article/details/76837708)

---

## `tee`

E.g.

```shell
cd pytorch
python setup.py install 2>&1 | tee build.log
```

**_Ref:_** [Github pytorch/pytorch: Makefile:140: recipe for target 'all' failed #17514](https://github.com/pytorch/pytorch/issues/17514#issuecomment-467995533)

---

## Install Ubuntu

### Build USB Flash

Using [**UNetbootin**](https://unetbootin.github.io/) to build USB Flash.

**Note:**

- rufus not support legacy
- UltraISO not support legacy

**_References:_**

- [知乎: Ubuntu 安装时遇到的一些问题](https://zhuanlan.zhihu.com/p/70758540)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Linux Tips

## Update `.barchrc`

```shell
. ~/.bashrc
```

Another way

```
source ~/.bashrc
```

**_References:_**

- [stackoverflow: How do I reload .bashrc without logging out and back in?](https://stackoverflow.com/questions/2518127/how-do-i-reload-bashrc-without-logging-out-and-back-in)

## Rename a directory

```shell
mv <oldname> <newname>
```

**_References:_**

- [ask ubuntu: How do I rename a directory via the command line?](https://askubuntu.com/questions/56326/how-do-i-rename-a-directory-via-the-command-line)

## `cd` into previous path

```shell
cd -
```

## Using command change Ubuntu download source<br>

```shell
sed -i s@http://archive.ubuntu.com/ubuntu/@<source you want to use>@g /etc/apt/sources.list
sed -i s@http://security.ubuntu.com/ubuntu/@<source you want to use>@g /etc/apt/sources.list
```

For example, change ubuntu 18.04 apt source to ali source.

```shell
sed -i s@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g /etc/apt/sources.list
sed -i s@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g /etc/apt/sources.list
```

[Ubuntu 原列表](http://wiki.ubuntu.org.cn/%E6%BA%90%E5%88%97%E8%A1%A8)

## Install Java on Ubuntu with apt-get

```
sudo apt-add-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
```

**_References:_**

- [Blog: Installing Apache Spark on Ubuntu 16.04](https://www.santoshsrinivas.com/installing-apache-spark-on-ubuntu-16-04/)

## Differences of `ctr + d`, `ctr + z` and `ctr + c`

- `ctr + d`: terminate input or exit the terminal or shell
- `ctr + z`: suspend foreground processes
- `ctr + c`: kill foregrousortednd processes

When you suspend a process, if you want to recover it, using `jobs` and `fg`.

**_References:_**

- [StackExchange Unix & Linux: If you ^Z from a process, it gets “stopped”. How do you switch back in?](https://unix.stackexchange.com/questions/109536/if-you-z-from-a-process-it-gets-stopped-how-do-you-switch-back-in)

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

**_References:_**

- [Blog: tar 压缩解压缩命令详解](https://www.cnblogs.com/jyaray/archive/2011/04/30/2033362.html)
- [Blog: linux tar (打包.压缩.解压缩)命令说明 | tar 如何解压文件到指定的目录？](http://www.cnblogs.com/52linux/archive/2012/0luarocks install cutorch3/04/2379738.html)

## Send/Copy files between _Windows_ and _Linux_

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

| Options |         whole          |                     Description                     |
| :-----: | :--------------------: | :-------------------------------------------------: |
|   -c    |       --continue       |    Continue getting a partially-downloaded file.    |
|   -O    | --output-document=FILE | Write documents to FILE (Change download file name) |

Download batch url from txt

```shell
wget -r <downloadlink.txt>
```

**_References:_**

- [Blog: wget 批量下载](http://tomrose.iteye.com/blog/1055640)
- [每天一个 linux 命令（61）：wget 命令](http://www.cnblogs.com/peida/archive/2013/03/18/2965369.html)
- [Computer Hope: Linux wget command](https://www.computerhope.com/unix/wget.htm)

## Inspect GPU state in unbuntu using `watch`

`watch -n 0.5 nvidia-smi`

**_References:_**

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

**_References:_**

- [ask ubuntu: What is the correct way to completely remove an application?](https://askubuntu.com/questions/187888/what-is-the-correct-way-to-completely-remove-an-application)

## Create a new user

1.  If you are a `root` user,
    ```sh
    adduser <username>
    ```
    If you are a `non-root` user,
    ```bash
    sudo adduser <username>
    ```
2.  Set a passward
    ```bash
    Set password prompts:
    Enter new UNIX password:
    Retype new UNIX password:
    passwd: password updated successfully
    ```
3.  Follow the prompts to set the new user's information. It is fine to accept the defaults to leave all of this information blank.
4.  Use `usermod` command to add the user into the `sudo` group, otherwise the user cannot use `sudo`

```bash
usermod -aG sudo <username>
```

**_References:_**

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

**_References:_**

- [Linux 中国开源社区: 为你的 Linux 应用创建 .desktop 文件](https://linux.cn/article-9199-1.html)

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

**_References:_**

- [stackExchange: How to keep a python script running when I close putty](https://unix.stackexchange.com/questions/362115/how-to-keep-a-python-script-running-when-i-close-putty)

## Ubuntu GUI show current folder path

Using shortcut `ctr` + `l`, the gui will show current folder path.

**_References:_**

- [Blog: ubuntu 16.04 LTS - 显示文件路径](https://blog.csdn.net/chengyq116/article/details/78631110)

## Show total number of files in folder

```shell
ls | wc -l
```

**_References:_**

- [stackoverflow: Count number of files within a directory in Linux? [closed]](https://stackoverflow.com/questions/20895290/count-number-of-files-within-a-directory-in-linux)
- [LinuxQuestion.org: How to find the total number of files in a folder](https://www.linuxquestions.org/questions/linux-newbie-8/how-to-find-the-total-number-of-files-in-a-folder-510009/)

## Install `.snap` software

```bash
sudo snap install <snap_file_name>
#
sudo snap install --dangerous <snap_file_name>
```

**_References:_**

- [Ubuntu 16.04 LTS 如何使用 Snap 软件包](https://www.sysgeek.cn/use-snap-packages-ubuntu-16-04/)
- [ask ubuntu: Snap install failure: error: cannot find signatures with metadata for snap](https://askubuntu.com/questions/822765/snap-install-failure-error-cannot-find-signatures-with-metadata-for-snap)

## Use `fd` instead of `find`

`fd` is faster than `find`

Install from [Github sharkdp/fd](https://github.com/sharkdp/fd)

Common used command can be found from [here](https://github.com/sharkdp/fd#tutorial)

**_References:_**

- [Github sharkdp/fd](https://github.com/sharkdp/fd)
- [Linux 中国: Linux 中一种友好的 find 替代工具](https://linux.cn/article-9767-1.html)

<!--  -->
<br>

---

<!--  -->

## `at`

Do somethings at specific time.

1. Write a shell script

   ```shell
   #!/bin/bash
   echo <your password> | sudo -S <command>
   ```

   **_Ref:_** [stackoverflow: Use sudo with password as parameter [closed]](https://stackoverflow.com/a/11955358/4636081)

2. Run

   ```shell
   at now + 1 minutes < <shell script>
   # at 2:30 PM 07/22/2019
   ```

**_References:_**

- [Computer Hope: Linux at, batch, atq, and atrm commands](https://www.computerhope.com/unix/uat.htm)

### Check job and Remove job

```shell
# Check all job
atq
>>> job_id job_time user
# Remove job
at -r <job id>
```

**_Ref:_** [RCSG website: at -r](http://rcsg-gsir.imsb-dsgi.nrc-cnrc.gc.ca/documents/advanced/node159.html)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Errors and Solutions

## Ubuntu error: 'apt-add-repository: command not found'<br>

```shell
apt-get install software-properties-common
```

<!--  -->
<br>

---

<!--  -->

## Ubuntu error: `Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)`<br>

```shell
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
```

<!--  -->
<br>

---

<!--  -->

## `System program problem detected` occur when Ubuntu starts.

```
sudo rm /var/crash/*
```

**_References:_**

- [ask ubuntu: Getting “System program problem detected” pops up regularly after upgrade](https://askubuntu.com/questions/133385/getting-system-program-problem-detected-pops-up-regularly-after-upgrade/369297)

<!--  -->
<br>

---

<!--  -->

## `Error mounting /dev/sdb1`

```bash
sudo ntfsfix /dev/sdb1
```

**_References:_**

- [StackExchange: Error mounting /dev/sdb1 at /media/ on Ubuntu 14.04 LTS](https://askubuntu.com/questions/586308/error-mounting-dev-sdb1-at-media-on-ubuntu-14-04-lts)

<!--  -->
<br>

---

<!--  -->

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

**_References:_**

- [stackoverflow: X11 forwarding request failed on channel 0](https://stackoverflow.com/questions/38961495/x11-forwarding-request-failed-on-channel-0)
- [Ask Xmodulo: How to fix “X11 forwarding request failed on channel 0”](http://ask.xmodulo.com/fix-broken-x11-forwarding-ssh.html)
- [StackExchange: ssh returns message “X11 forwarding request failed on channel 1”](https://unix.stackexchange.com/questions/111519/ssh-returns-message-x11-forwarding-request-failed-on-channel-1)

<!--  -->
<br>

---

<!--  -->

## Disable mouse middle button paste

Input the followings in terminal

```shell
xmodmap -e "pointer = 1 25 3 4 5 6 7 8 9"
```

If you want to set this forever, edit `~/.Xmodmap`

```shell
pointer = 1 25 3 4 5 6 7 8 9
```

**_References:_**

- [ubuntu 问答: 如何禁用鼠标中键点击粘贴？](https://ubuntuqa.com/article/687.html)
- [askubuntu: How do I disable middle mouse button click paste?](https://askubuntu.com/questions/4507/how-do-i-disable-middle-mouse-button-click-paste)

<!--  -->
<br>

---

<!--  -->

## Force reboot your Ubuntu desktop

Press `ctrl` + `alt` + `sys rq(printscreen)` and then do not loose `ctrl` and `alt`, and press `r`

**_References:_**

- [简书: Ubuntu 死机解决方法汇总](https://www.jianshu.com/p/36fb9eed82a3)

<!--  -->
<br>

---

<!--  -->

## Disable the `ublock your keyring` when you start Chrome

Open `Settings` -> `Details` -> `Users` -> Off `Automatic Login`

**_References:_**

- [askubuntu: How to disable the “unlock your keyring” popup?](https://askubuntu.com/questions/495957/how-to-disable-the-unlock-your-keyring-popup)

<!--  -->
<br>

---

<!--  -->

## Add `New Document` option in right click context menu

```bash
touch ~/Templates/Empty\ Document
```

**_References:_**

- [Blog: Add ‘New Document’ Option in Right Click Context Menu in Ubuntu 18.04 [Quick Tip]](https://itsfoss.com/add-new-document-option/)

<!--  -->
<br>

---

<!--  -->

## Restore `/home/<user>/Templates` when you delete it by mistake.

Create a new folder named `Templates` in `/home/<your user name>` and edit `~.config/user-dirs.dirs` like

```bash
XDG_TEMPLATES_DIR="$HOME/Templates"
```

**_References:_**

- [askubuntu: How do I restore the Templates folder in Ubuntu 18.04?](https://askubuntu.com/questions/1041732/how-do-i-restore-the-templates-folder-in-ubuntu-18-04)

<!--  -->
<br>

---

<!--  -->

## When log in, dock disapper (TODO)

Using `tweaks` and `dash to dock` to set `gnome-dock`. And using `alt` + `F2` and then press `r` to restart `gnome-shell`.

**_References:_**

- [Blog ubuntu 使用日常：18.04 中杂项处理 + 美化记录](https://hacpai.com/article/1527091030020)

<!--  -->
<br>

---

<!--  -->

## Ubuntu reinstall GUI

```bash
sudo apt update && sudo apt upgrade --fix-missing
```

```bash
sudo apt install --reinstall ubuntu-session gdm3
```

Ref [stackoverflow: GUI feature missing from full ubuntu installation 18.04](https://askubuntu.com/questions/1066048/gui-feature-missing-from-full-ubuntu-installation-18-04)

<!--  -->
<br>

---

<!--  -->

## Manage gcc version

Ref [Linux 公社: Ubuntu 18.04 下搭建 C/C++编译开发环境及 GCC 多版本切换](https://www.linuxidc.com/Linux/2019-04/158258.htm)

<!--  -->
<br>

---

<!--  -->

## Gnome

### Shortcut

- <code>Super + &#96;</code>: Switch betweent different window in same App.

**_Ref:_** [Linux 中国: 5 个提高效率的 GNOME 快捷键](https://mp.weixin.qq.com/s?__biz=MjM5NjQ4MjYwMQ==&mid=2664614460&idx=1&sn=5f51fe343eff38662c1015c98c7e91c6&chksm=bdcef17a8ab9786ccaae57f6768e2850f48d30091510904b06b751680280f4ded8059d4c034f&mpshare=1&scene=1&srcid=#rd)

<!--  -->
<br>

---

<!--  -->

## `mount` and `umount`

### Use `lsblk` to check all device on your compurter, and mount

```bash
mount /dev/<device name> <path you want to mount>
```

<!--  -->
<br>

---

<!--  -->

## Compress & Decompress

### `tar`

### `7za`

**Install**

```bash
# Ubuntu
sudo apt install p7zip-full
# CentOS
yum install p7zip
```

**_Ref:_** [stackoverflow: How can I uncompress a \*.7z file?](https://askubuntu.com/questions/219392/how-can-i-uncompress-a-7z-file)

**Decompress**

```bash
7za x <compress_file_name>
```

**_References:_**

- [CSDN: linux 下安装 7z 命令及 7z 命令的使用](https://blog.csdn.net/whatday/article/details/50157273)

### `zip`

- [RUNOOB.com: Linux zip 命令](https://www.runoob.com/linux/linux-comm-zip.html)

<!--  -->
<br>

---

<br>
<!--  -->

# ssh

## Generate ssh key

```shell
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

**_References:_**

- [Github Help: Generating a new SSH key and adding it to the ssh-agent](https://help.github.com/en/enterprise/2.16/user/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Common Software

- :thumbsup:[码农有道: 15 个超赞超牛逼的 Linux 工具，提高效率的同时增加乐趣！](https://mp.weixin.qq.com/s?__biz=MzIwNTc4NTEwOQ==&mid=2247487175&idx=2&sn=8c3e5b629b31c0c9232d14c73d5b4833&chksm=972adfbda05d56ab26a26ee1f51c4b8cb3d8d9f5be95cc9e782afffcfa4681017dc942605eec&mpshare=1&scene=1&srcid=&sharer_sharetime=1567609815220&sharer_shareid=cc363ceb86aab466ef27b423dcb88157#rd)

## Rufus

Ubuntu officially recommended tool to build bootable USB flash drives.

**_References:_**

- [CSDN: 在 Windows 中制作 Ubuntu 系统的 USB 启动盘](https://blog.csdn.net/u013553529/article/details/78307520)

<!--  -->
<br>

---

<!--  -->

## shadowsocks-qt-gui

```shell
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo apt-get update
sudo apt-get install shadowsocks-qt5
```

And you also need to install Chrome or Firefox extension `switchyomega`, you can get it from [github: FelisCatus/SwitchyOmega](https://github.com/FelisCatus/SwitchyOmega/releases). And the configuration can be get from its [wiki](https://github.com/FelisCatus/SwitchyOmega/wiki/GFWList).

<!--  -->
<br>

---

<!--  -->

## synergy

```bash
sudo apt-get install quicksynergy
```

<!--  -->
<br>

---

<!--  -->

## Filezilla

```bash
sudo apt-get install filezilla
```

### Config

### Set remote not show hidden file

Using `file filter` to realize this.

**_References:_**

- [FileZilla: Please do not show hidden file on SFTP](https://trac.filezilla-project.org/ticket/2685)

<!--  -->
<br>

---

<!--  -->

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

<!--  -->
<br>

---

<!--  -->

## Gitkraken

Good git client on Linux os.
You can get the `.deb` from [here](https://www.gitkraken.com/) , and use `dpkg -i <gitkraken.deb>` to install it.

### Problems & Solutions

#### [Error] Cannot start

If you double click the software icon, there is nothing happened. You'd better start the software by using command in terminal, and find what wrong during starting.

Ubuntu 18.04 can not start `Gitkraken`. Occur error about `libgnome-keyring.so.0: cannot open shared object file: No such file or directory`

**_References:_**

- [StackExchange: Error running Gitkraken even though dependencies installed](https://superuser.com/questions/1233459/error-running-gitkraken-even-though-dependencies-installed)

#### [Error] `Inotify Limit Error`

> Inotify (inode notify) is a Linux kernel subsystem that acts to extend filesystems to notice changes to the filesystem, and report those changes to applications.

```bash
su root
echo 99999 > /proc/sys/fs/inotify/max_user_watches
```

Refer [Tech Sparx Blog: How to fix GitKraken Inotify Limit Error - upgrading Ubuntu/Linux inotify limits](https://techsparx.com/blog/2018/02/gitkraken-inotify.html)

<!--  -->
<br>

---

<!--  -->

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
      <img src="https://upload-images.jianshu.io/upload_images/9890707-c26500c03483e88e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="80%">
    </p>

  Set Terminator path and parameters `--working-directory=%d/%b`

    <p align="center">
      <img src="https://upload-images.jianshu.io/upload_images/9890707-67e487b56b57153d.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="80%">
    </p>

- Set unlimited scroll in terminator

Teminator -> Preferences -> Profiles -> Scrolling and select inifinite scrollback.

**_References:_**

- [CSDN: linux 终端输出太多前面看不到的解决办法](https://blog.csdn.net/wuyu92877/article/details/74202163)
- [CSDN: Linux 终端内容太多无法全部显示](https://blog.csdn.net/wuyu92877/article/details/74202163)
- [stackoverflow: Unlimited Scroll in Terminator](https://askubuntu.com/questions/618464/unlimited-scroll-in-terminator)

**Some shortcuts**

- Split terminals horizontally: `Ctrl + Shift + o`
- Split terminals vertically: `Ctrl + Shift + E`
- Move to the terminal above the current one: `Alt + ↑`
- Move to the terminal below the current one: `Alt + ↓`
- Move to the terminal left the current one: `Alt + ←`
- Move to the terminal left the current one: `Alt + →`
- Open a new terminal in current window: `Ctrl + Shit + T`
- Net tab terminal: `Ctrl + PageUp` or `Ctrl + PageDown`

**_References_**

- [简书： 5 分钟入手 Terminator](http://www.jianshu.com/p/cee2de32ca28)
- [ubuntu: Setting nautilus-open-terminal to launch Terminator rather than gnome-terminal](https://askubuntu.com/questions/76712/setting-nautilus-open-terminal-to-launch-terminator-rather-than-gnome-terminal)
- [stackoverflow: Nautilus-actions in 18.04](https://askubuntu.com/questions/1030940/nautilus-actions-in-18-04)

<!--  -->
<br>

---

<!--  -->

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

**_References_**:

- [Github: robbyrussell/oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh)
- [LINUX 大棚： Linux 命令五分钟-用 chsh 选择 shell](http://roclinux.cn/?p=739)

### Plugins

#### zsh-syntax-highlighting

- Install
  1. cd into `~/.oh-my-zsh/plugins`
  2. `git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting`
  3. add `zsh-syntax-highlighting` into `~/.zshrc` `plugin`

**_References:_**

- [Github: zsh-users/zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting/blob/master/INSTALL.md)

### Problems & Solutions

- `tab completion not working`
  **_References:_**

  - [Github robbyrussell/oh-my-zsh Issue: tab completion not working #5651](https://github.com/robbyrussell/oh-my-zsh/issues/5651)

- `apt-get remove eclipse*` `*` not work, occur `no matches found: *`
  Edit `~/.zshrc` and add `setopt no_nomatch`
  **_References:_**
  - [Blog: zsh 不兼容的坑-zsh:no matches found](https://blog.csdn.net/u012675539/article/details/52079013)

<!--  -->
<br>

---

<!--  -->

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

**_Reference:_**

- [ubuntu 16.04 安装 matlab](http://blog.csdn.net/jesse_mx/article/details/53956358)

<!--  -->
<br>

---

<!--  -->

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
- CentOS
  Download cmake-3.3.2.rpm and cmake-gui-3.3.2 from [here](https://centos.pkgs.org/7/ghettoforge-plus-x86_64/cmake-gui-3.3.2-1.gf.el7.x86_64.rpm.html). Then use `rpm` tp install.

  **CentOS** using `yum` install `.rpm` and all all dependencies

  ```shell
  yum localinstall <rpm http link>
  ```

  **_References:_**

  - [stackoverflow: How to install rpm file along with its all dependencies?](https://superuser.com/questions/638616/how-to-install-rpm-file-along-with-its-all-dependencies)

**_References:_**

- [StackExchange-ask ubuntu: How to install cmake 3.2 on ubuntu 14.04?](https://askubuntu.com/questions/610291/how-to-install-cmake-3-2-on-ubuntu)
- [pkgs.org: cmake-gui-3.3.2-1.gf.el7.x86_64.rpm](https://centos.pkgs.org/7/ghettoforge-plus-x86_64/cmake-gui-3.3.2-1.gf.el7.x86_64.rpm.html)

<!--  -->
<br>

---

<!--  -->

## System Monitor

This software is default installed on Ubuntu. It makes inspecting `disk/cpu/memory` use condition simply.

<!--  -->
<br>

---

<!--  -->

## Netease cloud music(网易云音乐)

When start netease-cloud-music occur error `Local file: "" ("netease-cloud-music")`, you must start netease cloud music with `sudo`. One way to solve this trouble is to set `alias music="sudo netease-cloud-music"`

**_References:_**

- [知乎: Ubuntu 18.04 网易云音乐无法打开问题解决方案](https://zhuanlan.zhihu.com/p/37324458)

- [知乎: 在 Ubuntu 上有什么必装的实用软件？](https://www.zhihu.com/question/19811112)

<!--  -->
<br>

---

<!--  -->

## GlodenDict

Powerful dictionary on Linux.

```bash
sudo apt-get install goldendict
sudo apt-get install goldendict-wordnet
```

**_References:_**

- [开源中国: ubuntu 下安装 GoldenDict 替代有道词典--支持划词选词](https://my.oschina.net/u/1998467/blog/300643)

### Config

**_References:_**

- [简书: Linux 安装多功能词典 GoldenDict](https://www.jianshu.com/p/1af17b722991)

<!-- ***References:***

- [StarDict Dictionaries -- 星际译王词库 词典下载](http://download.huzheng.org/)
- [Ubuntu 14.04 安装配置强大的星际译王（stardict）词典](https://blog.csdn.net/tecn14/article/details/25917149) -->

<!--  -->
<br>

---

<!--  -->

## tmux

> tmux is a terminal multiplexer: it enables a number of terminals to be created, accessed, and controlled from a single screen. tmux may be detached from a screen and continue running in the background, then later reattached.

One useful function of `tmux` is **Keep run wihout exist when using ssh to remote server**

Ubuntu can install by `sudo apt-get install tmux`

### Tips

#### All tmux shortcuts need hit prefix `ctrl + b` and then\*\*

- `d`: detach (tmux still run)
- `[`: Scroll in tmux
  **_References:_**
  - [StackExchange: How do I scroll in tmux?](https://superuser.com/questions/209437/how-do-i-scroll-in-tmux)

**_References:_**

- [GithubL ryerh/tmux-cheatsheet.markdown](https://gist.github.com/ryerh/14b7c24dfd623ef8edc7)
- [Github: henrik/tmux_cheatsheet.markdown](https://gist.github.com/henrik/1967800)
- [Blog: Tmux 使用手册](http://louiszhai.github.io/2017/09/30/tmux/)

<!--  -->
<br>

---

<!--  -->

## vim

### Tips

- `find next`: `/` and then use `n` to the next

### Install

You'd better install Vim with `python` support. You can use `vim --version` to check if vim is installed with `python`

**Build** `vim` from [source](https://github.com/vim/vim/releases) with Python:

1.  Install `python-dev`

    - Ubuntu
      ```shell
      sudo apt-get install python3-dev
      ```
    - Centos
      `shell yum search python3 | grep devel`

          ```shell
          yum install python36-devel
          ```

      **_Referneces:_**

    - [ubuntu packages: python3-dev (3.5.1-3)](https://packages.ubuntu.com/xenial/python3-dev)
    - [stackoverflow: How to install python3-devel on red hat 7](https://stackoverflow.com/questions/43047284/how-to-install-python3-devel-on-red-hat-7)

2.  `cd` into vim folder

3.  Config

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
    --prefix=/usr/local \
    --enable-fail-if-missing
    ```

    When config `vim` with `python`, it need `python-config` or `python3-config`, you can run `python3-config --configdir` to get path of `python3-config`.

    > I would also recommend running configure with --enable-fail-if-missing so that the configure script will fail instead of quietly warning that it didn't find a python3 config directory or executable.

    **Note:** vim can be built with python2 and python3, but when install `Youcompleteme`, it need only one python version. So, when you build vim, you'd better only choose on python version. If you want to build vim with python2, you need to change `--enable-python3interp=yes` to `--enable-pythoninterp=yes` and `--with-python3-config-dir=~/software/anaconda/bin/` to `--with-python-config-dir=/usr/lib/python2.7/config`

    **_References:_**

    - [stackoverflow: VIM installation and Anaconda](https://stackoverflow.com/a/41917764/4636081)
    - [vim - configure for python3 support but still shows -python3 in version information](https://stackoverflow.com/a/26443517/4636081)

4.  `make -j8` and `sudo make install`

5.  Config `~/.vimrc`

    ```vim
    set nocompatible              " be iMproved, required
    filetype off                  " required

    set nu!
    syntax on
    set ts=4
    set backspace=2
    set expandtab

    " Youcompleteme
    set completeopt-=preview
    let g:ycm_add_preview_to_completeopt = 0

    " set the runtime path to include Vundle and initialize
    set rtp+=~/.vim/bundle/Vundle.vim
    call vundle#begin()
    " alternatively, pass a path where Vundle should install plugins
    "call vundle#begin('~/some/path/here')

    " let Vundle manage Vundle, required
    Plugin 'VundleVim/Vundle.vim'

    " rivergold install:
    Plugin 'Valloric/YouCompleteMe'

    " All of your Plugins must be added before the following line
    call vundle#end()            " required
    filetype plugin indent on    " required
    " To ignore plugin indent changes, instead use:
    "filetype plugin on
    "
    " Brief help
    " :PluginList       - lists configured plugins
    " :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
    " :PluginSearch foo - searches for foo; append `!` to refresh local cache
    " :PluginClean      - confirms removal of unused plugins; append `!` to    auto-approve removal
    "
    " see :h vundle for more details or wiki for FAQ
    " Put your non-Plugin stuff after this line
    ```

### Problems & Solutions

- Centos 7.2 install `python36-devel` occur error `Requires: libcrypto.so.10(openssl.1.0.2)(64bit)`
  You need to update openssl from `1.0.1` to `1.0.2`. You nend download [openssl-libs-1.0.2k-12.el7.x86_64.rpm](https://centos.pkgs.org/7/centos-x86_64/openssl-libs-1.0.2k-12.el7.x86_64.rpm.html) and run `rpm -i openssl-libs-1.0.2k-12.el7.x86_64.rpm`. And then it may occur another error like `conflicts with file from package`, you can use `rpm -i --replacefiles openssl-libs-1.0.2k-12.el7.x86_64.rpm`

  **_References:_**

  - [Blog: 解决 CentOS 下的 conflicts with file from 错误.](http://rayfuxk.iteye.com/blog/2280643)

### Plugins

When you want to enable or disable a plugin in vim, you need to add or change `Plugin '<plugin name>'` in `~/.vimrc`.

#### Vundle

Install

```shell
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```

#### Youcompleteme

You can install `Youcompleteme` using `vundle`, but it will take a long time. Here, we install it from source.

1. **Make sure** your vim is build with python2 or python3
2. Clone `Youcompleteme` source code into `~/.vim/bundle/`
   ```shell
   git clone --recursive https://github.com/Valloric/YouCompleteMe.git
   ```
3. `cd ~/.vim/bundle/YouCompleteMe`
   ```shell
   python ./install.py
   ```
   **Note:** If your vim is built with python2, here you need to use python2, if your vim is built with python3 here you need to use python3.

**\*References\*\***

- [Gith Valloric/YouCompleteMe](https://github.com/Valloric/YouCompleteMe#ubuntu-linux-x64)
- [简书: 一步一步带你安装史上最难安装的 vim 插件 —— YouCompleteMe](https://www.jianshu.com/p/d908ce81017a)

**Problems & Solutions**

- Using python3 run `python ./install.py` occur error `AttributeError: module 'enum' has no attribute 'IntFlag'`
  You need to uninstall `enum34` by run `pip uninstall enum34`
  **_References:_**

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

**_References:_**

- [Blog: ubuntu 下安装自动补全 YouCompleteMe](https://www.cnblogs.com/litifeng/p/6671446.html)
- [Github: Valloric/YouCompleteMe Issue: Auto close preview window after insertion not closing sometimes](https://github.com/Valloric/YouCompleteMe/issues/524)

<!--  -->
<br>

---

<!--  -->

## Vutrl VPS

### Install ssr

Get SSR install script from [here.](https://github.com/Alvin9999/new-pac/wiki/%E8%87%AA%E5%BB%BAss%E6%9C%8D%E5%8A%A1%E5%99%A8%E6%95%99%E7%A8%8B)

[Check if your vutrl vps IP is forbidden](https://www.vultrcn.com/11.html).

### Install git

Set server:

1. Install git: `yum install git`
2.

**_References:_**

- [Blog: 在 VPS 上搭建 Git 服务器](http://liujinlongxa.com/2016/08/07/%E5%9C%A8VPS%E4%B8%8A%E6%90%AD%E5%BB%BAGit%E6%9C%8D%E5%8A%A1%E5%99%A8/)
- [Shelton Blog: 搭建自己的 git 服务器](http://shelton13.github.io/2016/11/21/%E6%90%AD%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84git%E6%9C%8D%E5%8A%A1%E5%99%A8/)
- [Github LandChanning/DevNote 搭建 Git 服务器](https://github.com/LandChanning/DevNote/blob/master/20160716_%E6%90%AD%E5%BB%BAGit%E6%9C%8D%E5%8A%A1%E5%99%A8.md)

#### Somethings about `ssh`

`ssh` config file is in `/etc/ssh/sshd_config`

```shell
# Open rsa
RSAAuthentication yes
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
# Open password
PermitEmptyPasswords no
PasswordAuthentication yes
```

**注：** `ssh`可以设置成支持`ras`公钥秘钥的登录方式（需要你将自己电脑上的公钥放置到服务器的`authorized_keys`），同时也支持密码的登录

After you change the `ssh_config`, you need to restart `ssh` by `sudo service sshd restart`.

**_References:_**

- [CentOS: Securing OpenSSH](https://wiki.centos.org/HowTos/Network/SecuringSSH)
- [Blog: Vultr VPS SSH 密钥登录](http://zlxdike.github.io/2017/05/28/Vultr-VPS-SSH%E5%AF%86%E9%92%A5%E7%99%BB%E5%BD%95/)

**遇到的问题:** 在 Ubuntu 下使用`ctr` + `c`拷贝`rsa_pub`时，出现在粘贴后第一行的字母丢失的问题，暂时还未解决。

<!--  -->
<br>

---

<!--  -->

## Screen recorder: Kazam

Install

```bash
sudo apt-get install kazam
```

**_References:_**

- [Blog: 9 Best Screen Recorders For Linux](https://itsfoss.com/best-linux-screen-recorders/)
- [askubuntu: How to install Kazam 1.5.3?](https://askubuntu.com/questions/766440/how-to-install-kazam-1-5-3)

<!--  -->
<br>

---

<!--  -->

## FeedReader: RSS reader

Install FeedReader from [here](https://github.com/jangernert/FeedReader)

<!--  -->
<br>

---

<!--  -->

## Stretchly: Break time reminder app

Install stretchly from [here](https://github.com/hovancik/stretchly)

<!--  -->
<br>

---

<!--  -->

## Gnome Tweak Tool

Tool for ubuntu to config gnome

### Install

```bash
sudo apt-get install gnome-tweak-tool
```

**_References:_**

- [Linux Config: How to install Tweak Tool on Ubuntu 18.04 Bionic Beaver Linux](https://linuxconfig.org/how-to-install-tweak-tool-on-ubuntu-18-04-bionic-beaver-linux)

### Extentions

- `dash to dock`
  **_References:_**
  - [Dash to Dock GNOME Shell Extension](https://micheleg.github.io/dash-to-dock/download.html)

<!--  -->
<br>

---

<!--  -->

## LibreOffice

### Using `simsun` in LibreOffice

1. Prepare `simsun` and `微软雅黑`, you can download from [here](http://blog.51cto.com/geekz/716535)
2. Copy `.tff` into `/user/share/fonts` and run `fc-cache-fv` update font cache.

**_References:_**

- [Blog: Ubuntu 字体美化-微软雅黑和宋体](http://blog.51cto.com/geekz/716535)
- [LibreOffice: 适用于 LibreOffice 的中日韩字体](https://zh-cn.libreoffice.org/download/fonts/)

<!--  -->
<br>

---

<!--  -->

## flameshot

A screenshot software on Linux

### Install

```bash
sudo apt install  flameshot
```

### Config

Set a shortcut for flameshot

Open `Settings` -> `Devices` - > `Keyboard` -> Add new shotcut `Ctrl` + `Super` + `a`

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-5cce32c6eaf87f23.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="60%">
</p>

**_References:_**

- [Linux 中国: Flameshot：一个简洁但功能丰富的截图工具](https://linux.cn/article-10180-1.html)
- [Linux 中国: 在 Linux 下截屏并编辑的最佳工具](https://zhuanlan.zhihu.com/p/45919661)

<!-- ### shutter

A screenshot software on Ubuntu. And you can set keyboard shotcut for it.

***References:***

- [Linux公社: Ubuntu 安装截图工具Shutter，并设置快捷键 Ctrl+Alt+A](https://www.linuxidc.com/Linux/2015-07/119753.htm) -->

<!--  -->
<br>

---

<!--  -->

## VNC

Install VNC server on Remote PC and use VNC client to get desktop of remote pc at local.

### Install

1. Install VNC server on Remote

   ```shell
   yum install tigervnc-server
   ```

2. Install GNOME Desktop

   ```shell
   yum groupinstall -y "GNOME Desktop"
   ```

   Then, you need to `reboot` your remote computer.

3. Configure VNC Service

   ```shell
   cp /lib/systemd/system/vncserver@.service /etc/systemd/system/vncserver@:1.service
   ```

   And then `vim` `vncserver@:1.service`,

   ```vim
   32 [Unit]
   33 Description=Remote desktop service (VNC)
   34 After=syslog.target network.target
   35
   36 [Service]
   37 Type=forking
   38 # Clean any existing files in /tmp/.X11-unix environment
   39 ExecStartPre=/bin/sh -c '/usr/bin/vncserver -kill %i > /dev/null 2>&1 || :'
   40 ExecStart=/usr/sbin/runuser -l <USER> -c "/usr/bin/vncserver %i"
   41 PIDFile=/home/<USER>/.vnc/%H%i.pid
   42 ExecStop=/bin/sh -c '/usr/bin/vncserver -kill %i > /dev/null 2>&1 || :'
   43
   44 [Install]
   45 WantedBy=multi-user.target
   ```

   You need to set `<USER>` to your remote pc user, e.g. if your remote pc user is root, you need to set `<USER>` as `root`.

4. Make config effective

   ```shell
   systemctl daemon-reload
   ```

5. Set password

   ```shell
   vncpassed
   ```

6. Start VNC

   ```shell
   systemctl enable vncserver@:1.service #设置开机启动
   systemctl start vncserver@:1.service #启动vnc会话服务
   systemctl status vncserver@:1.service #查看nvc会话服务状态
   systemctl stop vncserver@:1.service #关闭nvc会话服务
   netstat -lnt | grep 590*      #查看端口
   ```

7. COnfigure remote pc firewall to allow `5901`

   ```shell
   firewall-cmd --state
   > running
   # If not running
   systemctl start firewalld
   ```

   Then allow `5901`

   ```shell
   firewall-cmd --permanent --zone=public --add-port=5901/tcp
   firewall-cmd --reload
   ```

8. Use VNC client to connect vnc

**_References:_**

- [Blog: CentOS7.2 安装 VNC，让 Windows 远程连接 CentOS 7.2 图形化界面](http://blog.51cto.com/12217917/2060252)
- [DigitalOcean: How To Install and Configure VNC Remote Access for the GNOME Desktop on CentOS 7](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-vnc-remote-access-for-the-gnome-desktop-on-centos-7)

<!--  -->
<br>

---

<!--  -->

## gnome-tweaks

Free customization and settings manager for the GNOME desktop.

**_References:_**

- [LINUXCONFIG.org How to install Tweak Tool on Ubuntu 18.04 Bionic Beaver Linux](https://linuxconfig.org/how-to-install-tweak-tool-on-ubuntu-18-04-bionic-beaver-linux)

<!--  -->
<br>

---

<!--  -->

## Lepton

[Lepton][lepton home] is a lean code snippet manager based on GitHub Gist.

Support:

- macOS
- Win
- Linux

[lepton home]: https://github.com/hackjutsu/Lepton

<!--  -->
<br>

---

<!--  -->

## redshift

Adjusts the color temperature of your screen

[Redshift Home](https://github.com/jonls/redshift)

Ubuntu install

```bash
sudo apt install redshift-gtk
```

<!--  -->
<br>

---

<!--  -->

## nmon

Performance Monitor for Linux.

### Install

TBD

### Use

- `q`: Exit
- `h`: help
- `r`: System information
- `c`: CPU
- `m`: Memory
- `d`: Disk
- `k`: Kernel
- `n`: Network
- `j`: file system
- `t`: Top-processes

Ref [Linux 中国: 使用 Nmon 监控 Linux 的系统性能](https://linux.cn/article-6886-1.html)

<!--  -->
<br>

---

<!--  -->

## zssh & rzsz

### Install

- Ubuntu: `sudo apt install zssh`

Ref [博客园: linux 机器之间 zssh, rz, sz 互相传输](https://www.cnblogs.com/strikebone/p/3454679.html)

### Use

Remote server (usually CentOS), need install `lrzsz`

```bash
yum install lrzsz
```

Ref [CSDN: linux CentOS 安装 rz 和 sz 命令 lrzsz](https://blog.csdn.net/jack85986370/article/details/51321475)

#### Upload

1. `ctr + @`: 切换到 local
2. `sz <file>`: 从 local 上传文件到 remote

#### Download

1. `sz <file>` 在 remote 启动 sz，准备发送文件
2. `ctr + @`: 切换到 local
3. `cd <path>`: 进入到需要存储的路径
4. `rz`: 接受文件

**_References:_**

- [Blog: zssh 使用](https://phenix3443.github.io/notebook/ubuntu/zssh.html)

<!--  -->
<br>

---

<!--  -->

## rTorrent

Linux download Torrent tool.

### Install

```bash
# Ubuntu
sudo apt install rTorrent
# CentOS
yum install rTorrent
```

**_Ref:_** [TecMint: 5 Linux Command Line Based Tools for Downloading Files and Browsing Websites](https://www.tecmint.com/linux-command-line-tools-for-downloading-files/)

### Use

**_Ref:_** [Linux 中国: rtorrent - 强大的命令行 BT 客户端](https://linux.cn/article-2064-1.html)

<!--  -->
<br>

---

<!--  -->

## ag

Faster than grep and ack

### Install

**Ubuntu**

```shell
sudo apt install silversearcher-ag
```

**_References:_**

- [Ming's Blog: the silver searcher：快如闪电的 grep](https://bitmingw.com/2017/10/08/fast-grep-the-silver-searcher/)

### Usage

**_References:_**

- [Ming's blog: the silver searcher：快如闪电的 grep](https://bitmingw.com/2017/10/08/fast-grep-the-silver-searcher/)

<!--  -->
<br>

---

<!--  -->

## tig

Very good tool for command git

### Install

**Ubuntu**

```shell
sudo apt install tig
```

**_References:_**

- [掘进: tig — 让 git 命令行可视化](https://juejin.im/post/5afabcb1f265da0b886d92b6)

<!--  -->
<br>

---

<!--  -->

## Proxychains

- [Github](https://github.com/rofl0r/proxychains-ng)

### Install

1. Clone source

   ```shell
   git clone https://github.com/rofl0r/proxychains-ng.git
   ```

2. Build

   ```shell
   cd proxychains-ng
   ./configure --prefix=/usr --sysconfdir=/etc
   make
   make install
   make install-config
   cd .. && rm -rf proxychains-ng
   ```

3. Config

   Edit `/etc/proxychains.conf`

   ```conf
   socks5 <ip> <port>
   ```

**_References:_**

- [Harker' Blog: Centos 7 安装 Proxychains 实现 Linux 代理](http://www.harker.cn/archives/proxychains.html)

---

## Eclipse

```shell
sudo apt install default-jre
sudo snap install --classic eclipse
```

**_References:_**

- [stackoverflow: Eclipse doesn't start on Ubuntu 18.04](https://askubuntu.com/questions/1031171/eclipse-doesnt-start-on-ubuntu-18-04)

---

## snap

Snap download is very slow.

1. Download snap file from [uApp Explorer](https://uappexplorer.com/snap/ubuntu/eclipse)
2. Install via `sudo snap install xxx.snap --dangerous`

**_References:_**

- [CSDN: ubuntu 使用 snap 下载速度慢](https://blog.csdn.net/qf0129/article/details/91873740)

---

## mpv

Good video player on Linux.

### Config

Edit `~/.config/mpv/mpv.conf`

#### Window size

```shell
geometry=50%x50%
```

**_References:_**

- [Github mpv-player/mpv: Want to start mpv in a default window size and position #4724](https://github.com/mpv-player/mpv/issues/4724)
