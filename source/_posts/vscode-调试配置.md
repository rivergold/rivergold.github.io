---
title: vscode - 调试配置
categories: Tech
tags:
  - vscode
abbrlink: eca434c1
date: 2022-03-11 10:25:39
---

最近在工作中使用C++开发框架，期间需要进行debug，所以学习了下如何在vscode中配置相关编译和调试的配置。

## vscode配置c++开发环境

vscode使用三种文件实现对c++开发的环境管理、编译和调试

- `c_cpp_properties.json`：C++开发环境的管理
- `tasks.json`：编译过程的配置
- `launch.json`：调试配置

### 熟悉vscode c++配置中的常用的变量

- ${workspaceFolder}：当前工作区目录

### 创建c_cpp_properties.json

快捷键：command + p，选择> C/C++: Edit Configuration(UI)

<p align="center">
<img src="https://s2.loli.net/2022/03/19/EkzDtRbAQGOFL9q.png" width="80%" height="80%"
</p>

根据提示进行配置

<p align="center">
<img src="https://s2.loli.net/2022/03/19/cuvtnR91GWhBHVo.png" width="80%" height="80%"
配置c_cpp
</p>

### 创建tasks.json

快捷键：command+p，选择>Tasks: Configure Task -> Create tasks.json file from template -> Others

<p align="center">
<img src="https://s2.loli.net/2022/03/19/CwJ1kTpE4v3GxyR.png" width="60%" height="60%"/>
</p>

<p align="center">
<img src="https://s2.loli.net/2022/03/19/2qcIo6J74kgulbV.png" width="60%" height="60%"
根据template创建task
</p>

```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "echo",
            "type": "shell",
            "command": "echo Hello"
        }
    ]
}
```

修改为

```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "run_cmake",
            "type": "shell",
            "command": "mkdir -p ./build; cd ./build; cmake -DCMAKE_BUILD_TYPE=Debug ..; make -j8;",
            "group": {
                "kind": "build",
                "isDefault": true
            }
        },
        {
            "label": "clean",
            "type": "shell",
            "command": "make clean",
        }
    ]
}
```

vscode使用tasks实现与外界功能的交互，有点像Gitlab CI的配置

- label：当前task的名称
- group：当前task所属的组

### 创建launch.json

点击vscode左侧的Run and Debug图标，选择create a launch.json file

<p align="center">
<img src="https://s2.loli.net/2022/03/19/jAvT9aJeY2Pq83D.png" width="80%" height="80%"
创建launch_json
</p>

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(lldb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "enter program name, for example ${workspaceFolder}/a.out",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "lldb"
        }
    ]
}
```

修改为

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(lldb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/run_main",
            "preLaunchTask": "run_cmake", // 依赖之前的run_cmake, 每次debug时会
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}", // 运行debug的路径
            "environment": [],
            "externalConsole": false,
            "MIMode": "lldb"
        }
    ]
}
```

References:

- [简书：vscode + cmake编译环境配置](https://www.jianshu.com/p/ad29eee7b736)
