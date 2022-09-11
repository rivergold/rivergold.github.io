---
title: 搭建Gitlab CI
categories: Tech
tags:
  - Git
  - Docker
abbrlink: 3053651376
date: 2022-08-26 14:46:00
---


因组内算法开发工作需要搭建一套Gitlab CI实现算法底库的自动化编译和Linux端/车机端离线测试，这里记录一下踩过的坑🤣

## 简介

GitLab CI/CD 是一个内置在GitLab中的工具，用于通过持续方法进行软件开发：

- Continuous Integration (CI)  持续集成
- Continuous Delivery (CD)     持续交付
- Continuous Deployment (CD)   持续部署

持续集成的工作原理是将小的代码块推送到Git仓库中托管的应用程序代码库中，并且每次推送时，都要运行一系列脚本来构建、测试和验证代码更改，然后再将其合并到主分支中。

持续交付和部署相当于更进一步的CI，可以在每次推送到仓库默认分支的同时将应用程序部署到生产环境。

这些方法使得可以在开发周期的早期发现bugs和errors，从而确保部署到生产环境的所有代码都符合为应用程序建立的代码标准。

GitLab CI/CD 由一个名为 .gitlab-ci.yml 的文件进行配置，改文件位于仓库的根目录下。文件中指定的脚本由GitLab Runner执行。

## 基于Docker搭建Gitlab CI

为了快速部署，我采用的方案是在将Gitlab Runner和Excutor都部署在docker中。

这里Gitlab Runner的工作原理为：runner运行在一个docker container中，其负责维护一个对应repo的基础分支，当有作业任务分发到当前runner时，runner会根据CI yaml所配置的image创建并启动一个对应的container，在这个container中切换分支更新代码后执行相应的CI命令。

### 安装Gitlab Runner

> GitLab Runner is an application that works with GitLab CI/CD to run jobs in a pipeline.

Gitlab Runner是Gitlab CI/CD实现运行各种作业的软件应用。所以想让自己的机器可以与Gitlab交互运行Gitlab上配置的作业任务，第一步需要在机器上安装Gitlab Runner。安装Gitlab Runner可以直接下载软件安装，也可以使用官方发布的Docker Image，这里我们第二种方法。

```shell
docker pull gitlab/gitlab-runner
```

注意：这里有个天坑，老版本的gitlab-runner（<=14.xx）无法控制Runner上接收的job数量，比如配置了Runner仅能同时跑一个job，但是老版本的Runner控制错误导致多个job同时跑在同一个runner的同一个excutor中，具体的问题可以在[Issue: Concurrent jobs on a single Runner sometimes run in the same CI_PROJECT_DIR](https://gitlab.com/gitlab-org/gitlab-runner/-/issues/3688)和[Issue: CI-Runner shell executor runs concurrently with concurrency=1](https://gitlab.com/gitlab-org/gitlab-runner/-/issues/4793)，Gitlab-Runner开发组应该是2022年初解决的这个问题。

### 启动Gitlab Runner

{% tabs 启动Gitlab Runner %}
<!-- tab Linux -->
```bash
docker run -d --name gitlab-runner --restart always \
  -v /srv/gitlab-runner/config:/etc/gitlab-runner \
  -v /var/run/docker.sock:/var/run/docker.sock \
  gitlab/gitlab-runner:latest
```
<!-- endtab -->
<!-- tab Windows -->
```bash
docker run -d --name gitlab_runner-gwalglib  --restart always -v C:\gitlab-runner\config:/etc/gitlab-runner -v /var/run/docker.sock:/var/run/docker.sock gitlab/gitlab-runner:latest
```
<!-- endtab -->
{% endtabs %}

- `/etc/gitlab-runner`: gitlab-runner的配置文件保存路径

### 注册Runner

上一步我们启动了runner程序，下一步需要将runner注册到对应的Gitlab Repo中，只有注册后Repo的作业任务才能分发到runner程序上。

使用命令`docker exec -it gitlab-runner /bin/bash`进入到我们刚刚启动的GitLab Runner的container中，执行注册命令

```shell
gitlab-runner register
```

根据交互提示，完成runner注册。

### 修改配置

配置文件的路径在`/etc/gitlab-runner/config.toml`

```shell
concurrent = 2
check_interval = 0

[session_server]
  session_timeout = 1800

[[runners]]
  name = "your_runner_name"
  url = "your_url"
  id = 1
  token = "xxx"
  token_obtained_at = xxx
  token_expires_at = xxx
  executor = "docker"
  limit = 2
  [runners.custom_build_dir]
  [runners.cache]
    [runners.cache.s3]
    [runners.cache.gcs]
    [runners.cache.azure]
  [runners.docker]
    tls_verify = false
    image = "python:latest"
    privileged = false
    disable_entrypoint_overwrite = false
    oom_kill_disable = false
    disable_cache = false
    volumes = ["/cache"]
    shm_size = 0
    pull_policy = ["if-not-present"]
    network_mode = "host"
```

- `concurrent`: 并行数，同时可以运行的job数量
- `limit`: runner任务队列中允许接收的job数量
- `pull_policy = ["if-not-present"]`: 如果使用的docker image已经存在在机器上，那不会执行pull
- `network_mode = "host"`: docker使用host的网络

完成配置修改后，Gitlab Runner就配置完成啦~

## Gitlab Yaml Config

```yaml
default:
  image:
    name: <默认使用的docker image>

stages:
  - code_analysis
  - build
  - test
  - auto_test
  - parse_auto_test

variables:
  # 使用私有的docker仓库，在执行完docker login之后，可以在`~/.docker/config.json`中找到
  DOCKER_AUTH_CONFIG: '{"auths": {"<域名>": {"auth": "<授权token>"}}}'

before_script:
  - git submodule sync --recursive
  - git submodule foreach --recursive git fetch
  - git submodule update --init
  - git lfs pull

clang-format:
  stage: code_analysis
  script:
    - python3 CI/CodeAnalysis/FormatCheck/run-clang-format.py --style Google -r ./
  tags:
    - cpp_build # 使用对应tag的Runner
  only:
    - merge_requests

linux:build:
  stage: build
  script:
    - bash CI/ci-linux-build.sh
  artifacts:
    paths:
      - build/
    exclude:
      - build/install/**/*.pt # 去除pt文件
  tags:
    - cpp_build
  # only:
  #   - merge_requests
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $schedule_linux_auto_test_memory == "True"

linux:test:
  stage: test
  needs:
    job: "linux:build"
    artifacts: true
  script:
    - bash CI/ci-run-linux-test.sh
  tags:
    - test
  only:
    - merge_requests
  retry:
    max: 1
    when: always

# ------
# [Linux] Memory test
# ------
# Memory test
linux:auto_test-memory:
  stage: auto_test
  needs:
    job: "linux:build"
    artifacts: true
  artifacts:
    paths:
      - build/auto_test_result/
  script:
    - bash CI/ci-linux-auto-test.sh
  tags:
    - autotest
  only:
    refs:
      - schedules
    variables:
      - $schedule_linux_auto_test_memory == "True"
  retry:
    max: 1
    when: always

# Memory report
linux:auto_test_report-memory:
  stage: parse_auto_test
  before_script:
    - which python
  needs:
    job: "linux:auto_test-memory"
    artifacts: true
  artifacts:
    paths:
      - build/auto_test_result/
  # variables:
  #   DOCKER_HOST: "https://hub.docker.com/"
  image:
    name: ubuntu_20_04-python:v1.0
  script:
    - python tools/ProcessAutoTestMemoryData.py build/auto_test_result/memory_collect_data.txt
  tags:
    - autotest
  only:
    refs:
      - schedules
    variables:
      - $schedule_linux_auto_test_memory == "True"
  retry:
    max: 1
    when: always

# ------
# [Car] Memory test
# ------
# Memory test
car:auto_test-memory:
  stage: auto_test
  needs:
    job: "qnx:build"
    artifacts: true
  image:
    name: <对应的docker image>
  variables:
    ADB_IP: $CI_ADB_IP # 使用Gitlab界面配置的Variable值
    CAR_ANDROID_DIR: "/data/vendor/nfs/mount/fota/auto_test"
    CAR_QNX_DIR: "/usr/nfs_ota/fota/auto_test"
  before_script:
    - git lfs pull
  script:
    - chmod a+x ./CI/ci-qnx-auto-test/test_memory/ci_run_test_memory.sh
    - ./CI/ci-qnx-auto-test/test_memory/ci_run_test_memory.sh
  artifacts:
    paths:
      - auto_test-car_result/
  tags:
    - autotest
  only:
    refs:
      - schedules
    variables:
      - $schedule_car_auto_test_memory == "True"
  retry:
    max: 1
    when: always
```

References:

- [stackoverflow: Gitlab DOCKER_AUTH_CONFIG not working](https://stackoverflow.com/a/65810302)
