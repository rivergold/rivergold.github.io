# 虚拟机与GPU使用
- [Jarvis and GPU](http://jarvis.gitlab.qiyi.domain/jarvis/)
- [iqiyi-wiki: 虚拟主机(VPS)服务手册](http://wiki.qiyi.domain/pages/viewpage.action?pageId=6818391)

# Linux
## Linux查看硬盘分区
```
df -lh
```

***References:***
- [Linux 查看磁盘分区、文件系统、使用情况的命令和相关工具介绍](http://blog.51cto.com/13233/82677)

## CentOS
### `rpm`的使用
- [Blog: CentOS RPM 包使用详解](http://blog.51cto.com/tengq/1930181)

# Tools
## Using `pip` install **OpenCV**
- [Medium: Installing OpenCV 3.3.0 on Ubuntu 16.04 LTS](https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961)


# Frameworks
## Torch
### Install
- [torch: Getting started with Torch](http://torch.ch/docs/getting-started.html)

When CentOS install torch, it occur error: **xxxx**

when install torch error: https://ubuntuforums.org/showthread.php?t=1670531

### Problems and Solutions
- Install [torch-opencv](https://github.com/VisionLabs/torch-opencv) ouucurs errors: `Cannot Find OpenCV` or other errors about OpenCV
    **Solution:** Change `CMakeLists.txt` and set OpenCV path. Then use `luarocks make` to build and install package.
    ***References:*** 
    - [isionLabs/torch-opencv: Problem installing from source](https://github.com/VisionLabs/torch-opencv/issues/182)


## Deep Learning Model Convertor
- [Github: ysh329/deep-learning-model-convertor](https://github.com/ysh329/deep-learning-model-convertor)