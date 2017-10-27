# Base Linux Command
- How to update .barchrc(update bash path)?([ref][1])<br>
    ```shell
    . ~/.bashrc
    ```

- How to rename a directory?([ref][2])<br>
    ```shell
    mv <oldname> <newname>
    ```

- How `cd` into previous path?<br>
    ```shell
    cd -
    ```

- Using command change Ubuntu download source<br>
    ```shell
    sed -i s@http://archive.ubuntu.com/ubuntu/@https://mirrors.ustc.edu.cn/ubuntu/@g /etc/apt/sources.list
    ```

- Ubuntu error: 'apt-add-repository: command not found'<br>
    ```
    apt-get install software-properties-common
    ```

- [How to install Java on Ubuntu with apt-get](https://www.digitalocean.com/community/tutorials/how-to-install-java-on-ubuntu-with-apt-get)

- Ubuntu error: `Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)`<br>
    ```shell
    sudo rm /var/cache/apt/archives/lock
    sudo rm /var/lib/dpkg/lock
    ```

- Differences of `ctr + d`, `ctr + z` and `ctr + c`<br>
    - `ctr + d`: terminate input or exit the terminal or shell
    - `ctr + z`: suspend foreground processes
    - `ctr + c`: kill foreground processes

- Change `pip` and `conda` donload source
    - `pip`
        1. Create a folder named `.pip` in `~/`
        2. Create a file named `pip.conf`
        3. Write the followings into `pip.conf`
        ```
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

- How can *Windows client* visit *Linux Host* and send/copy files?
    - Install SSH Clients on windows client computer
    - Using `ssh` can visit and log remote linux host
    - `pscp` command can send/download files to/from remote linux host
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

- `wget` command
    ```shell
    wget [options] <url>
    ```

    | Options |  whole  |                Description                         |
    |:--:|:------------:|:--------------------------------------------------:|
    | -c | --continue | Continue getting a partially-downloaded file. |

    - References
        - [每天一个linux命令（61）：wget命令](http://www.cnblogs.com/peida/archive/2013/03/18/2965369.html)
        - [Computer Hope: Linux wget command](https://www.computerhope.com/unix/wget.htm)

- How to see GPU state in unbuntu ([\*ref](https://unix.stackexchange.com/questions/38560/gpu-usage-monitoring-cuda))
    - Run `watch -n 0.5 nvidia-smi` in terminal

[1]:http://stackoverflow.com/questions/2518127/how-do-i-reload-bashrc-without-logging-out-and-back-in
[2]:http://askubuntu.com/questions/56326/how-do-i-rename-a-directory-via-the-command-line

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
2. Go to `System Settings` -> `Language Support`, change `Keyboard input method system as `fcitx`.
3. Download Sogou input method from [here](https://pinyin.sogou.com/linux/?r=pinyin)
```
sudo dpkg -i install <sogou.deb>
```
