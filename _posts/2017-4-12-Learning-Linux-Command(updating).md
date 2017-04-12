- How to update .barchrc(update bash path)?([ref][ref_1])<br>
    ```shell
    . ~/.bashrc
    ```

- How to rename a directory?([ref][ref_2])<br>
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

[ref_1]: http://stackoverflow.com/questions/2518127/how-do-i-reload-bashrc-without-logging-out-and-back-in

[ref_2]:
http://askubuntu.com/questions/56326/how-do-i-rename-a-directory-via-the-command-line
