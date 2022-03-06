---
title: Git Memo - 常用命令
tags:
- git
---

## Generate Key

生成ssh所需要的key

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

## `git config`

对git进行配置，配置可以作用于当前repo，也可以配置为全局的。常用的有配置git的username和email。

### 显示当前的username和email

```bash
git config user.name
git config user.email
```

### 配置当前repo的username和email

```bash
# name
git config user.name
# email
git config user.email
```

### 配置全局的username和email

```bash
# username
git config --global user.name "<your name>"
# email
git config --global user.email "<your email>"
```

## `git clone`

从远端仓库通过http或者ssh协议克隆仓库到本地

最基础的操作为

```bash
git clone <remote_url> [本地目录]
```

## `git branch`

用于对分支的查看、创建和删除

### 查看分支

```bash
# 查看本地分支，当前分支前会有*号
git branch
# 查看远端分支
git branch -r
# 查看所有分支，包括本地和远端
git branch -a
```

### 创建分支

```bash
# 创建分支，只创建不会切换分支
git branch <new_branch_name>
```

### 删除分支

```bash
# 删除分支，如果分支没有被merge，会删除失败
git branch -d <branch_name>
# 强制删除分支
git branch -D <branch_name>
```

## `git checkout`

实现对分支的切换，同时也可以创建分支、放弃对某个文件的修改

对分支的基本操作

```bash
# 切换到分支
git checkout <branch_name>
# 切换到分支，如果分支不存在则创建
git checkout -b <branch_name>
```

`git checkout -b <branch_name>`等同于`git branch <branch_name>` + `git checkout <branch_name>`

对文件的基本操作

```bash
# 放弃对单个文件的修改
git checkout <filename>
# 放弃对当前目录下的修改
git checkout .
```

### 切换到某个tag

```bash
git checkout <tag_name>
```

### 从某个tag创建分支

```bash
git checkout tags/<tag_name> -b <new_branch_name>
```

### 从远端分支创建新的分支

```bash
# 方法1
git checkout <remote_name>/<remote_branch_name> -b <local_branch_name>
# 方法2
git checkout -t <remote_name>/<branch_name>
```

## `git fetch`

## `git pull`

## `git stash`

将当前工作区的内容暂存起来

### git stash with custom message

```shell
git stash serve "your message"
```

[stackoverflow: How to name and retrieve a stash by name in git?](https://stackoverflow.com/questions/11269256/how-to-name-and-retrieve-a-stash-by-name-in-git)

## `git rebase`

```bash
git checkout branch-a # 切换到branch-a
git rebase branch-b # rebase a on top of b，将branch a换基到b上
```

git rebase的命令含义为：将 `branch-a` 搬移到 `branch-b` 上，即这条命令改变的是`branch-a`，对`branch-a`进行了换基操作。
此时，`LOCAL`为`branch-a`，`REMOTE`为`branch-a`

**切记：如果你想向公共的分支（大家都会使用的，该分支会接受多人的 commit）提交 commit，只能使用 merge 的方式。即：不要在公共的分支上执行`git rebase xxx`， only rebase private branches。**
因为在公共的分支上执行`git rebase xxx`会导致公共分支的 commit 记录发生变化，从而导致别人的公共分支和你的不一样，该问题会很严重。
因此，在 Github 中如果你想给开源仓库的 master 提交代码，需要提 pull request，当 request 被同意时，Github 会采用 pull 的方式将你的提交 merge 到主分支上。

有两个比较好的网站内容介绍了rebase的使用，值得推荐：

:thumbsup: [TOWER: Rebase 代替合并](https://www.git-tower.com/learn/git/ebook/cn/command-line/advanced-topics/rebase)
:thumbsup: [stackoverflow: Git merge master into feature branch](https://stackoverflow.com/a/16956062/4636081)

<!-- ### 解决rebase的冲突 -->

## 确定`LOCAL`和`REMOTE`

在进行`merge`或者`rebase`时，如果出现冲突，显示和解决冲突时需要分辨`LOCAL`和`REMOTE`

进行merge时

```bash
git checkout A
git merge    B    # merge B into A
```

`LOCAL`为`A`
`REMOTE`为`B`

进行`rebase`时

```bash
git checkout A
git rebase   B    # rebase A on top of B
```

`LOCAL`为`B`
`REMOTE`为`A`

总结，作为“主干”的分支是`LOCAL`，“外来”的分支为`REMOTE`

:thumbsup: [stackoverflow: git rebase, keeping track of 'local' and 'remote'](https://stackoverflow.com/questions/3051461/git-rebase-keeping-track-of-local-and-remote/3052118#3052118)

## References

- [Github Help: Generating a new SSH key and adding it to the ssh-agent](https://help.github.com/en/enterprise/2.18/user/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
- [stackoverflow: Download a specific tag with Git](https://stackoverflow.com/questions/791959/download-a-specific-tag-with-git)
- [stackoverflow: How do I check out a remote Git branch?](https://stackoverflow.com/a/1783426/4636081)
- [博客园：Git branch && Git checkout常见用法](https://www.cnblogs.com/qianqiannian/p/6011404.html)
