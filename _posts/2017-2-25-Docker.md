<!-- # _Docker_ -->
This is a tutorial about base operation of Docker.
## Common command:
- List all installed images.<br>
    ```
    docker images
    ```

- List all containers.<br>
    ```
    docker ps -a
    ```

- Stop all containers.<br>
    ```
    docker stop $(docker ps -a -q)
    ```
    -q(--quiet), Only display numeric IDs.

- Remove a container.<br>
    ```
    docker rm <contrain name>
    ```

- Remove all containers.<br>
    ```
    docker rm $(docker -a -q)
    ```

- Remove a image.<br>
    ```
    docker rmi <image name and tag>
    ```

- Run a container which is already exiting.<br>
    ```
    docker start <container_id or name>
    docker exec -ti <container_id or name> /bin/bash
    ```

- Copy file from host into container in shell (From [stackoverflow][ref_1]).<br>
    ```
    docker cp <file name> container:<path>
    ```

- Convert container into image.<br>
    ```
    docker commit <container name/id> <image name>
    ```
    It is used to build new image from container.

- Save docker image into host disk.<br>
    ```
    docker save -o <path you want to save> <image name and tag>
    ```
    often save docker image as .tar

- Load docker image from disk.<br>
    ```
    docker load -i <image file path>
    ```

## Valuble Docker Image
- [dl-docker](https://github.com/floydhub/dl-docker):
    all-in-one docker image for deep learning.


## Problem and Solution
- How to share fold betweent host and container on Docker for Windows? [(ref)][ref_2]
    - Open Docker for Windows
    - Set `Shared Drives`
    - run docker contrainer with `-v [fold path on host]:[fold path on contrainer]`

- How to use jupyter notebook in docker? localhost:8888 not work? [(ref)][ref_3]
    - _Solution:_ The ip of container in docker is 0.0.0.0, but default ip address in jupyter is 127.0.0.1. So we should change jupyter notebook ip if we want to use it on our host computer. Input `jupyter note --ip=0.0.0.0` in your docker container and then open localhost:8888 in your browser, and see it will work ok.

## Tips
- Useful command in ubnutu image:
    - Change ubuntu download source, using `sed`<br>
        ```
        sed -i s@<needed replace content>@<replace content>@g <file path>
        ```

        E.g. ```s@http://archive.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g /etc/apt/sources.list```<br>
        After you change the source list, you need to update it to let it work via `sudo apt-get update`

[ref_1]:http://stackoverflow.com/questions/22907231/copying-files-from-host-to-docker-container
[ref_2]:https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c#.8tny4uf9o
[ref_3]:https://github.com/gopherds/gophernotes/issues/6
