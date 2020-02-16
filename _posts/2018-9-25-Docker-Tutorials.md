<!-- # _Docker_ -->

This is a tutorial about base operation of Docker.

# Common Commands:

## List all installed images

```bash
docker images
```

## List all containers

```bash
docker ps -a
```

## Stop all containers

```bash
docker stop $(docker ps -a -q)
```

`-q(--quiet)`: Only display numeric IDs.

## Remove a container

```bash
docker rm <contrain name>
```

## Remove all containers.<br>

```bash
docker rm $(docker ps -a -q)
```

## Remove a image

```bash
docker rmi <image name and tag>
```

## Run a container which is already exiting

```bash
docker start <container_id or name>
docker exec -ti <container_id or name> /bin/bash
```

## Copy file from host into container in shell

```bash
docker cp <file name> container:<path>
```

**_References:_**

- [Stackoverflow: Copying files from host to Docker container](https://stackoverflow.com/questions/22907231/copying-files-from-host-to-docker-container)

## Convert container into image

It is used to build new image from container.

```bash
docker commit <container name/id> <image name>
```

## Save docker image into host disk

Often save docker image as .tar

```
docker save -o <path you want to save> <image name and tag>
```

## Load docker image from disk

```
docker load -i <image file path>
```

<!--  -->
<br>

---

<!--  -->

# Tips & Tricks

## Install Docker CE for Ubuntu

- [docker docs: Get Docker CE for Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)

## Set docker run path.

Default path is /var/lib/docker, it is not good for images and container to be in `/`. The solution is to add `data-root` to `/etc/docker/daemon.json`

```bash
{
    "data-root": <path>,
}
```

**Note:** Line in `.json` must be end with `,` excpect for the last line.

**_Reference:_**

- [Github-moby/moby: Deprecate --graph flag; Replace with --data-root #28696](https://github.com/moby/moby/pull/28696)
- [docker docs: Configure the Docker daemon](https://docs.docker.com/engine/admin/#configure-the-docker-daemon)
- [archlinux: Docker](https://wiki.archlinux.org/index.php/Docker)

## Set docker accelerator with ali yun

1. Get your docker accelerator address from your [Container Hub](https://cr.console.aliyun.com/) in Ali yun.
2. Add `registry-mirrors` to `/etc/docker/daemon.json`
   ```bash
   "registry-mirrors": ["<your accelerate address>"],
   ```
3. Reload and restart `docker`
   `bash sudo systemctl daemon-reload sudo systemctl restart docker`
   **_References:_**

- [阿里云栖社区： Docker 镜像加速器](https://yq.aliyun.com/articles/29941)

## Run docker without `sudo`

1. Create the `docker` group
   ```bash
   sudo groupadd docker
   ```
2. Add your user to the `docker` group
   ```bash
   sudo usermod -aG docker $USER
   ```
3. Log out and log back

**_References:_**

- [docker docs: Manage Docker as a non-root user](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user)

## Using **nvidia-docker** run cuda in docker container.

Install **nvidia-docker** from [Github-NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and create new container with `nvidia-docker`<br>

**Problems and Solutions**:

- `Error: Could not load UVM kernel module. Is nvidia-modprobe installed?`
  Install `nvidia-modprobe`([\*ref](https://askubuntu.com/questions/841824/how-to-install-nvidia-modprobe))<br>

  ```shell
  sudo apt-add-repository multiverse
  sudo apt update
  sudo apt install nvidia-modprobe
  ```

- `Error response from daemon: create nvidia_driver_352.63: create nvidia_driver_352.63: Error looking up volume plugin nvidia-docker: plugin not found.`
  1. Check the status of the plugin using via `service nvidia-docker status` or `systemctl status nvidia-docker`
  2. run `sudo nvidia-docker-plugin` in shell
  3. restart docker `sudo restart docker`
  4. logout
  5. test `nvidia-docker run --rm nvidia/cuda nvidia-smi` again

## Change Ubuntu download source with command, using `sed`<br>

```bash
sed -i s@<needed replace content>@<replace content>@g <file path>
```

E.g.

```shell
sed -i s@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g /etc/apt/sources.list
```

After you change the source list, you need to update it to let it work via `sudo apt-get update`

**_References:_**

- [Ubuntu 中文 Wiki: 下载源](https://wiki.ubuntu.com.cn/%E6%BA%90%E5%88%97%E8%A1%A8)

## How to use jupyter notebook in docker? localhost:8888 not work?

The ip of container in docker is 0.0.0.0, but default ip address in jupyter is 127.0.0.1. So we should change jupyter notebook ip if we want to use it on our host computer. Input `jupyter note --ip=0.0.0.0` in your docker container and then open localhost:8888 in your browser, and see it will work ok.

**_References:_**

- [Github-gopherdata/gophernotes](https://github.com/gopherdata/gophernotes/issues/6)

## When delete image, error `Can’t delete image with children occur.

Use `docker inspect --format='{{.Id}} {{.Parent}}' $(docker images --filter since=<image_id> -q)`. And then delete sub_image using `docker rmi {sub_image_id}`

## [Windows] Using python in Docker Linex container, has error like `UnicodeEncodeError: 'ascii' codec can't encode character '\u22f1' in position 242`

```python
import sys
sys.stdout
>>> <_io.TextIOWrapper name='' mode='w' encoding='ANSI_X3.4-1968'>
```

Change output encode as utf-8

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

**_References:_**

- [解决 Python3 下打印 utf-8 字符串出现 UnicodeEncodeError 的问题](https://www.binss.me/blog/solve-problem-of-python3-raise-unicodeencodeerror-when-print-utf8-string/)

## Run gui in docker container on docker for Ubuntu

```bash
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
```

It may occur error:

```shell
No protocol specified
QXcbConnection: Could not connect to display :0
[1]    6407 abort (core dumped)  cmake-gui
```

Need run `xhost +local:docker`

**_References:_**

- [DOCKER COMMUNITY FORUMS: Start a GUI-Application as root in a Ubuntu Container](https://forums.docker.com/t/start-a-gui-application-as-root-in-a-ubuntu-container/17069)

## [Windows] How run linux gui in docker container on docker for windows?

1. Install **Cygwin** with **Cygwin/x** on your computer.
2. In cygwin terminal, run
   ```shell
   export DISPLAY=<your-machine-ip>:0.0
   startxwin -- -listen tcp &
   xhost + <your computer ip>
   ```
3. In your powershell, run
   `bash docker run --it -e DISPLAY=<your computer ip>:0.0 <image> /bin/bash`
   **Problem & Solution**

- Error: `xhost: unable to open display`(\*[ref](https://forums.freebsd.org/threads/50613/))
  Use `rm ~/.Xauthority`, then try again previous steps.

**_References_**

- [Blog: Running a GUI application in a Docker container](https://linuxmeerkat.wordpress.com/2014/10/17/running-a-gui-application-in-a-docker-container/)
- [Running Linux GUI Apps in Windows using Docker](https://manomarks.net/2015/12/03/docker-gui-windows.html)
- [Stackoverflow: How to run GUI application from linux container in Window using Docker?](http://stackoverflow.com/questions/29844237/how-to-run-gui-application-from-linux-container-in-window-using-docker)
- [Linux: Running GUI apps with Docker](http://fabiorehm.com/blog/2014/09/11/running-gui-apps-with-docker/)

## [Windows] On docker for Windows, how host computer powershell direct ping to container(container ip is 172.17.0.1)?

Run docker container with `--net=<net bridge>`, set a net bridge for container.

## [Windows] How to share fold betweent host and container on Docker for Windows?

1. Open Docker for Windows
2. Set `Shared Drives`
3. run docker contrainer with `-v [fold path on host]:[fold path on contrainer]`

**_References:_**

- [Romin Irani’s Blog: Docker on Windows — Mounting Host Directories](https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c)

## Docker container connect to host via port `8080`

`docker run` with `-P`

By default, docker container's 8080 port doesn's connect to any port of host, so container cannot access to host. `-P` will ask Docker to give container a random port to the host.

**_Ref:_** [Docker-从入门到实践: 映射容器端口到宿主主机的实现](https://yeasy.gitbooks.io/docker_practice/advanced_network/port_mapping.html)

<!--  -->
<br>

---

<!--  -->

# Dockerfile

## Command

### `ENV`

Edit `path` in dockerfile

```bash
ENV PATH $PATH:~/software/anaconda/bin
```

**_References:_**

- [stackoverflow: condas `source activate virtualenv` does not work within Dockerfile](https://stackoverflow.com/questions/37945759/condas-source-activate-virtualenv-does-not-work-within-dockerfile)

### `RUN`

- `RUN <command>`: the command is run in a shell, which by defalut is `/bin/sh -c`
- `RUN ["executable", "param1", "param2"]`

So, if you use `RUN source ~/.bashrc`, if will occur **error** `source: not found`, because it uses `/bin/sh` not `/bin/bash`. One solution is to set `bash` as default shell.

```docker
SHELL ["/bin/bash", "-c"]
```

**_Referneces:_**

- [stackoverflow: Using the RUN instruction in a Dockerfile with 'source' does not work](https://stackoverflow.com/questions/20635472/using-the-run-instruction-in-a-dockerfile-with-source-does-not-work)
- [Github moby/moby Issues: How to make builder RUN use /bin/bash instead of /bin/sh #7281](https://github.com/moby/moby/issues/7281)
- [docker docs: run](https://docs.docker.com/engine/reference/builder/#run)

- `RUN cd <path>`: it only work at current command.
  ```dockerfile
  # miniconda.sh in ~/software
  RUN cd ~/software
  RUN bash miniconda.sh
  > error: cannot find miniconda.sh
  ```

### `COPY`

`COPY [src] [dist]`: The dist path in docker container must be absolute path.

**Copy folder from host into container in shell:**

```docker
ADD go /usr/local/go
# or
COPY go /usr/local/go
```

**_Ref:_** [stackoverflow: Copy directory to other directory at Docker using ADD command](https://stackoverflow.com/a/26504961/4636081)

## Tips

### `echo` string into file with line break

```bash
echo -e "\n<string>\n"
```

**_References:_**

- [stackoverflow: Echo newline in Bash prints literal \n](https://stackoverflow.com/questions/8467424/echo-newline-in-bash-prints-literal-n)
- [Blog: linux echo 命令的-n、-e 两个参数](http://blog.sina.com.cn/s/blog_4da051a6010184uk.html)

### Install `anaconda` in dockerfile

```docker
# Method 1: From web and download install pack
RUN ["/bin/bash", "-c", "wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O <path>/miniconda.sh"]
RUN ["/bin/bash", "-c", "$HOME/miniconda.sh -b -p <path>/anaconda"]

# Method 2: Copy anaconda/miniconda.sh and install
COPY ./Miniconda3-latest-Linux-x86_64.sh /root/software
RUN cd /root/software &&  /bin/bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /root/software/anaconda

# Add conda into PATH
ENV PATH="$HOME/anaconda/bin:$PATH"
```

**Note** When I use `COPY ./Miniconda3-latest-Linux-x86_64.sh ~/software`, docker failed to copy file into `/root/software` and `bash` cannot find the file. But **absolute path** worked！

**_References:_**

- [conda docs: Installing in silent mode](https://conda.io/docs/user-guide/install/macos.html#install-macos-silent)
- [Gtihub: faircloth-lab/conda-recipes](https://github.com/faircloth-lab/conda-recipes/blob/master/docker/centos6-conda-build/Dockerfile)

## Problems & Solutions

### Dockerfile with CentOS `yum install xxx` occur error `Rpmdb checksum is invalid: dCDPT(pkg checksums): xxxx.amzn1`

```docker
RUN yum instal -y yum-plugin-ovl
```

**_References:_**

- [AWS Discussion Forums: Rpmdb checksum is invalid on yum install, amazonlinux as a docker base](https://forums.aws.amazon.com/thread.jspa?threadID=244745)
- [ORACLE: 2.6 Overlayfs Error with Docker](https://docs.oracle.com/cd/E93554_01/E69348/html/uek4-czc_xmc_xt.html)
- [Github moby/moby: overlayfs fails to run container with a strange file checksum error #10180](https://github.com/moby/moby/issues/10180#issuecomment-378005800)

### `COPY ../installer/Miniconda3-latest-Linux-x86_64.sh /root/software` occur error `COPY failed: Forbidden path outside the build context: ../installer/Miniconda3-latest-Linux-x86_64.sh`

Please use absolute path.

## My dockerfile

```docker
FROM nvidia/cuda:8.0-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]
# ADD ./sources.list /etc/apt/
RUN sed -i s@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g /etc/apt/sources.list

RUN apt-get update && apt-get install -y wget git

RUN mkdir ~/software
# ============================
# Install zsh
# ============================
RUN apt-get -y install zsh
RUN chsh -s $(which zsh)
# RUN sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
COPY ./oh-my-zsh-install.sh /root/software
RUN cd /root/software && bash oh-my-zsh-install.sh && rm -rf oh-my-zsh-install.sh
RUN sed -i s@"ZSH_THEME=\"robbyrussell\""@"ZSH_THEME=\"ys\""@g ~/.zshrc
SHELL ["/bin/zsh", "-c"]
RUN source ~/.zshrc


# ===================
# Install miniconda
# ===================
# RUN ["/bin/bash", "-c", "wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/software/miniconda3.sh"]
# RUN cd ~/software && /bin/bash ./miniconda3.sh -b -p ~/software/anaconda
ADD ./Miniconda3-latest-Linux-x86_64.sh /root/software
RUN cd ~/software && /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/software/anaconda && rm -rf Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/software/anaconda/bin:${PATH}"

# ===================
# Config pip
# ===================
RUN mkdir ~/.pip && touch ~/.pip/pip.conf
RUN echo -e "[global]\ntrusted-host=mirrors.aliyun.com\nindex-url=http://mirrors.aliyun.com/pypi/simple" >> ~/.pip/pip.conf
RUN pip install --upgrade pip

# ============================
# python packages
# ============================
RUN pip install numpy
```

### Tips

- `sed -i s@"ZSH_THEME=\"robbyrussell\""@"ZSH_THEME=\"ys\""@g ~/.zshrc`
  `sed -i s@ZSH_THEME="robbyrussell"@ZSH_THEME="ys"@g ~/.zshrc` not work, if `"` in source or replace string.

- `source ~/.zshrc` must use `zsh`, so need to set `SHELL ["/bin/zsh", "-c"]` before.

- Dockerfile change `$PATH`
  `ENV PATH=<path>:$PATH`
  **_References:_**
  - [DOCKER COMMUNITY FORUMS: Change \$PATH in ubuntu container so that it is accessible from outside the shell](https://forums.docker.com/t/change-path-in-ubuntu-container-so-that-it-is-accessible-from-outside-the-shell/19817/2)
  - [stackoverflow: In a Dockerfile, How to update PATH environment variable?](https://stackoverflow.com/a/38742545/4636081)

<!--  -->
<br>

---

<!--  -->

# Docker Introduction

- [Docker Toolbox, Docker Machine, Docker Compose, Docker WHAT!?](https://nickjanetakis.com/blog/docker-toolbox-docker-machine-docker-compose-docker-wtf)
- [Docker Explained](https://www.digitalocean.com/community/tutorials/docker-explained-using-dockerfiles-to-automate-building-of-images)

[ref_1]: http://stackoverflow.com/questions/22907231/copying-files-from-host-to-docker-container
[ref_2]: https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c#.8tny4uf9o
[ref_3]: https://github.com/gopherds/gophernotes/issues/6

<!--  -->
<br>

---

<!--  -->

# Valuble Docker Images

- [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/)
- [dl-docker](https://github.com/floydhub/dl-docker): all-in-one docker image for deep learning.

# Docker Images

## Ubuntu 16.04

### Some Problems

- If you want to build vim 8.0 with python3.6, you need install `python3.6-dev`. But Ubuntu 16.04 not have this package.

  ```shell
  sudo add-apt-repository ppa:deadsnakes/ppa
  ```

  **_References:_**

  - [stackoverflow: Why can't I install python3.6-dev on Ubuntu16.04](https://stackoverflow.com/questions/43621584/why-cant-i-install-python3-6-dev-on-ubuntu16-04)
  - [vsupalov: Developing With Python 3.6 on Ubuntu 16.04 LTS - Getting Started and Keeping Control](https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/)

<!--  -->
<br>

---

<!--  -->
