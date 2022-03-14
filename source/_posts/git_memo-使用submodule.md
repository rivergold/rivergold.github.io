---
title: Git Memo - 使用submodule
date: 2022-03-14 17:02:00
tags:
- git
---

当你所开发的repo需要使用另外一个repo时，你可以使用submodule对需要使用的repo进行管理。

## 为当前repo添加submodule

```bash
git submodule add <url> <path> # 将repo作为submodule添加到path中
git submodule update --init # 获取submodule的内容
# git submodule update --init --recursive
git commit -m "Add submodule xxx at path xxx"
```

`--recursive`的作用是，不仅仅获取submodule的内容，同时还会递归的获取submodule's submodule的内容。

在添加submodule后，git会创建/修改.gitmodules文件，同时修改.git/config文件：

```bash
# .gitmodules
[submodule "third_party/googletest"]
    ignore = dirty
    path = third_party/googletest
    url = https://github.com/google/googletest.git
# .git/config
[submodule "third_party/googletest"]
    active = true
    url = https://github.com/google/googletest.git
```

在不做修改的情况下，.gitmodules和.git/config关于submodule的配置是一样的，他们之间的区别在于：

- `.gitmodules`：对所有开发者的配置
- `.git/config`：本地配置，仅对你自己生效

## 克隆带有submodule的repo

克隆带有submodule的repo时，有两种方法：一种为先克隆仓库，之后获取子模块；另一种为在克隆仓库的同时获取子模块。

### 先克隆repo，后获取submodule

```bash
git clone <url>
```

查看相关submodule

```bash
$ git submodule
-e2239ee6043f73722e7aa812a459f54a28552929 third_party/googletest
-96046b8ccfb8e6fa82f6b2b34b3d56add2e8849c third_party/onnx
```

示例仓库包含了两个submodule，分别是googletest和onnx

（实际这里拿PyTorch仓库作为实验，它里面还有很多submodule，感兴趣的同学可以进一步了解）

`-`表示该submodule还没有被拉去到本地，是个空文件夹
`third_party/googletest`表示该submodule在repo中的路径

获取submodule

```bash
git submodule init [submodule路径] # 初始化本地配置文件
git submodule update # 获取submodule最新的内容
```

例如上述例子中的`thrid_party/googletest`和`third_party/onnx`的submodule都没有被配置，执行以下命令可以单独获取googletest的代码，而不拉去onnx的代码。

```bash
$ git submodule init third_party/googletest
Submodule 'third_party/googletest' (https://github.com/google/googletest.git) registered for path 'third_party/googletest'
$ git submodule update
Submodule path 'third_party/googletest': checked out 'e2239ee6043f73722e7aa812a459f54a28552929'
```

### 克隆repo同时获取submodule

```bash
git clone <url> --recurse-submodules
```

如果没有加`--recursive`参数，克隆到本地的repo中的submodule文件夹是空的。

## 查看submodule的状态

```bash
git submodule status
```

## 更新submodule

拉去submodule最新的提交，你可以cd到submodule的目录中，之后执行git pull，这个过程和你操作普通的repo是一样。另外，如果你不想手动进入到submodule目录下执行拉去，你可以使用`git submodule update --remote`命令

```bash
git submodule update --remote [submodule路径]
```

如果没有指定submodule路径，git将拉取所有submodule，对他们的master/main分支进行更新。

## 修改submodule追踪的分支

需要说明的一个知识点：git submodule默认追踪的是submodule的master/main分支，如果你想修改submodule追踪其它分支，你可以修改.gitmodules文件（该方法会让其他开发者采用相同的配置），也可以只修改本地的`.git/config`配置。

```bash
# 仅修改本地，该命令会修改.git/config文件
git config -f .gitmodules submodule.<path>.branch <branch_name>
# 修改使得对所有开发者生效，需要修改改.gitmodules文件；该操作也会同时修改.git/config
git config -f .gitmodules submodule.<path>.branch <branch_name>
```

这里给一个简单的示例：

```bash
# 从googletest远端拉去最新提交
$ git submodule update --remote third_party/googletest
Submodule path 'third_party/googletest': checked out 'ae1b7ad4308249bfa928e65d1a33be117fc0992c' # 此时本地仓库会显示third_party/googletest Modified
# 回退修改
git submodule update third_party/googletest
```

```bash
# 切换使用googletest的其他分支
git config submodule.third_party/googletest.branch dinord-patch-1
```

此时检查.git/config文件，会发现其被修改为

```bash
# 修改前
# [submodule "third_party/googletest"]
#     active = true
#     url = https://github.com/google/googletest.git

# 修改后
[submodule "third_party/googletest"]
    active = true
    url = https://github.com/google/googletest.git
    branch = dinord-patch-1
```

增加了一行branch = dinord-patch-1，表明配置googletest子目录默认追踪dinord-patch-1分支。如果想放弃修改，使用main作为默认分支，删除这一行就可以了。

执行带有`-f .gitmodules`参数的命令

```bash
# 切换使用googletest的其他分支
git config -f .gitmodules submodule.third_party/googletest.branch dinord-patch-1
```

会修改.gitmodules文件，再提交修改后会使得配置对所有开发者生效。

[git: 7.11 Git 工具 - 子模块](https://git-scm.com/book/zh/v2/Git-%E5%B7%A5%E5%85%B7-%E5%AD%90%E6%A8%A1%E5%9D%97)

## 更新submodule的url

当你所使用的submodule变更了url（比如你fork了一份开源代码到公司内部的gitlab中，并想将项目开发所需要的submodule切换到这个fork的repo上），你需要使用`git submodule sync`

```bash
git submodule sync --recursive # 将新的url复制到本地配置中
```

## 对每个submodule执行git命令

基本的使用方法为

```bash
git submodule foreach "<你想要执行的git命令>"
```

### 放弃对submodule的修改

```bash
git submodule foreach "git reset --hard"
```

[stackoverflow: How do I revert my changes to a git submodule?](https://stackoverflow.com/a/23668025)

## 删除不再使用的submodule

```bash
# step1: 删除.gitmodules中对应的行
# step2: 删除.git/config中对应的行
# step3: 删除submodule文件夹
rm -rf submodule-dir/
# 提交commit
git add
git commit -m "Remove submodule"
```
