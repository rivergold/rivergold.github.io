# Linux
## Linux查看硬盘分区
```
df -lh
```

***References:***
- [Linux 查看磁盘分区、文件系统、使用情况的命令和相关工具介绍](http://blog.51cto.com/13233/82677)

## `Error mounting /dev/sdb1`
***References:***
- [StackExchange: Error mounting /dev/sdb1 at /media/ on Ubuntu 14.04 LTS](https://askubuntu.com/questions/586308/error-mounting-dev-sdb1-at-media-on-ubuntu-14-04-lts)

## `ssh -X <user_name>@<ip>` occur error: `X11 forwarding request failed on channel 0
`
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

***References:***
- [stackoverflow: X11 forwarding request failed on channel 0](https://stackoverflow.com/questions/38961495/x11-forwarding-request-failed-on-channel-0)
- [Ask Xmodulo: How to fix “X11 forwarding request failed on channel 0”](http://ask.xmodulo.com/fix-broken-x11-forwarding-ssh.html)
- [StackExchange: ssh returns message “X11 forwarding request failed on channel 1”](https://unix.stackexchange.com/questions/111519/ssh-returns-message-x11-forwarding-request-failed-on-channel-1)


## CentOS
### `rpm`的使用
- [Blog: CentOS RPM 包使用详解](http://blog.51cto.com/tengq/1930181)

<!--  -->
<br>

***
<!--  -->

# Tools
## Using `pip` install **OpenCV**
- [Medium: Installing OpenCV 3.3.0 on Ubuntu 16.04 LTS](https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961)


<!--  -->
<br>

***
<!--  -->


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

### Lua
> Lua arrays begin with 1

## Deep Learning Model Convertor
- [Github: ysh329/deep-learning-model-convertor](https://github.com/ysh329/deep-learning-model-convertor)

# Tensorflow
## tf.nn, tf.layers, tf.contrib区别
***References:***
- [小鹏的专栏: tf API 研读1：tf.nn，tf.layers， tf.contrib概述](https://cloud.tencent.com/developer/article/1016697)

<!--  -->
<br>

***
<!--  -->

# Python
## Numpy
### Copy in numpy
***References:***
- [CSDN:【Python】numpy中的copy问题详解](https://blog.csdn.net/u010099080/article/details/59111207)

### `np.newaxis` add new axis to array
```
x = np.array([1, 2, 3, 4])
x = x[:, np.newaxis]
print(x)
>>> array([[1],
           [2],
           [3],
           [4],
           [5]])
x = np.array(range(4)).reshape(2, 2)
x = x[..., np.newaxis]
print(x.shape)
>>> (2, 2, 1)
```

***References:***
- [stackoverflow: How does numpy.newaxis work and when to use it?](https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it)


## OpenCV
### When OpenCV read color image, color information is stored as BGR, in order to convert to RGB
```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(<img_path>) # BRG
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```

***References:***
- [PHYSIOPHILE: WHY OPENCV USES BGR (NOT RGB)](https://physiophile.wordpress.com/2017/01/12/why-opencv-uses-bgr-not-rgb/)