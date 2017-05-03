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


[1]:http://stackoverflow.com/questions/2518127/how-do-i-reload-bashrc-without-logging-out-and-back-in
[2]:http://askubuntu.com/questions/56326/how-do-i-rename-a-directory-via-the-command-line
