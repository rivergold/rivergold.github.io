---
title: Linux Memo - 命令行挂载U盘
categories: Tech
tags:
  - Linux
abbrlink: 1bcaf7bc
date: 2022-03-05 11:55:21
---


使用命令行挂载U盘需要3步：

1. 找到插入的U盘的设备名
2. 挂载，之后使用
3. 安全弹出

## Step1 找到U盘的设备名

找到插入的U盘的设备名有很多种方法，rivergold推荐的最简单和直接的方法为：

```bash
dmesg # 根据输出信息查看设备名
```

## Step2 挂载

```bash
mkdir /media/xxx/xxx
sudo mount /dev/sdx1 /media/xxx/xxx
```

## Step3 安全弹出

```bash
udisksctl umount -b /dev/sdx1
udiskctl power-off -b /dev/sdx
```

References：

- [博客园: Linux系统下查看USB设备名及使用USB设备](https://www.cnblogs.com/rusking/p/6107989.html)
- [Stackoverflow: What is the Command Line Equivalent of "Safely Remove Drive"?](https://askubuntu.com/a/532691)
