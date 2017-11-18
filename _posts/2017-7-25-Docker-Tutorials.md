<!-- # _Docker_ -->
This is a tutorial about base operation of Docker.
# Common Commands:
## List all installed images
```
docker images
```

## List all containers
```
docker ps -a
```

## Stop all containers
```
docker stop $(docker ps -a -q)
```
`-q(--quiet)`: Only display numeric IDs.

## Remove a container
```
docker rm <contrain name>
```

## Remove all containers.<br>
```
docker rm $(docker -a -q)
```

## Remove a image
```
docker rmi <image name and tag>
```

## Run a container which is already exiting
```
docker start <container_id or name>
docker exec -ti <container_id or name> /bin/bash
```

## Copy file from host into container in shell
```
docker cp <file name> container:<path>
```
***References:***
- [Stackoverflow: Copying files from host to Docker container](https://stackoverflow.com/questions/22907231/copying-files-from-host-to-docker-container)

## Convert container into image
It is used to build new image from container.
```
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

<br>
<br>

# Tips & Tricks
## Install Docker CE for Ubuntu
[docker docs: Get Docker CE for Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)

## Set docker run path.
Default path is /var/lib/docker, it is not good for images and container to be in `/`. The solution is to add `data-root` to `/etc/docker/daemon.json`
```
{
    "data-root": <path>,
}
```
**Note:** Line in `.json` must be end with `,` excpect for the last line.
***Reference:***
- [Github-moby/moby: Deprecate --graph flag; Replace with --data-root #28696](https://github.com/moby/moby/pull/28696)
- [docker docs: Configure the Docker daemon](https://docs.docker.com/engine/admin/#configure-the-docker-daemon)
- [archlinux: Docker](https://wiki.archlinux.org/index.php/Docker)

## Set docker accelerator with ali yun
1. Get your docker accelerator address from your [Container Hub](https://cr.console.aliyun.com/) in Ali yun.
2. Add `registry-mirrors` to `/etc/docker/daemon.json`
    ```
    "registry-mirrors": ["<your accelerate address>"],
    ```
3. Reload and restart `docker`
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```
***References:***
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

***References:***
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
```
sed -i s@<needed replace content>@<replace content>@g <file path>
```
E.g.
```shell
sed -i s@http://archive.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g /etc/apt/sources.list
```
After you change the source list, you need to update it to let it work via `sudo apt-get update`

***References:***
- [Ubuntu中文Wiki: 下载源](https://wiki.ubuntu.com.cn/%E6%BA%90%E5%88%97%E8%A1%A8)

## How to use jupyter notebook in docker? localhost:8888 not work?
The ip of container in docker is 0.0.0.0, but default ip address in jupyter is 127.0.0.1. So we should change jupyter notebook ip if we want to use it on our host computer. Input `jupyter note --ip=0.0.0.0` in your docker container and then open localhost:8888 in your browser, and see it will work ok.

***References:***
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

***References:***
- [解决Python3下打印utf-8字符串出现UnicodeEncodeError的问题](https://www.binss.me/blog/solve-problem-of-python3-raise-unicodeencodeerror-when-print-utf8-string/)

## Run gui in docker container on docker for Ubuntu
```
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix 
```


## [Windows] How run linux gui in docker container on docker for windows?
1. Install **Cygwin** with **Cygwin/x** on your computer.
2. In cygwin terminal, run
    ```shell
    export DISPLAY=<your-machine-ip>:0.0
    startxwin -- -listen tcp &
    xhost + <your computer ip>
    ```
3. In your powershell, run
    ```
    docker run --it -e DISPLAY=<your computer ip>:0.0 <image> /bin/bash
    ```
**Problem & Solution**
- Error: `xhost:  unable to open display`(\*[ref](https://forums.freebsd.org/threads/50613/))  
    Use `rm ~/.Xauthority`, then try again previous steps.

***References***
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

***References:***
- [Romin Irani’s Blog: Docker on Windows — Mounting Host Directories](https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c)

<br>
<br>

# Docker Introduction
- [Docker Toolbox, Docker Machine, Docker Compose, Docker WHAT!?](https://nickjanetakis.com/blog/docker-toolbox-docker-machine-docker-compose-docker-wtf)
- [Docker Explained](https://www.digitalocean.com/community/tutorials/docker-explained-using-dockerfiles-to-automate-building-of-images)

[ref_1]:http://stackoverflow.com/questions/22907231/copying-files-from-host-to-docker-container
[ref_2]:https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c#.8tny4uf9o
[ref_3]:https://github.com/gopherds/gophernotes/issues/6

<br>
<br>

# Valuble Docker Images
- [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/)
- [dl-docker](https://github.com/floydhub/dl-docker): all-in-one docker image for deep learning.

<br>
<br>