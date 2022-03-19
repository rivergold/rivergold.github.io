---
title: Git Memo
date: 2022-02-27 11:23:35
tags:
- git
---

Some basic git usages and tricks.

## :fallen_leaf:Good Tutorials

- [Pro Git 中文版](https://bingohuang.gitbooks.io/progit2/content/)
- [GitHub 漫游指南](http://github.phodal.com/)

**_References:_**

- [知乎: 一些关于 Git 的学习资源](https://zhuanlan.zhihu.com/p/32379998?utm_source=ZHShareTargetIDMore&utm_medium=social&utm_oi=37839882420224)
- [知乎: 如何在 GitHub 上做一个优秀的贡献者？](https://www.zhihu.com/question/310488111/answer/585336948?utm_source=ZHShareTargetIDMore&utm_medium=social&utm_oi=37839882420224)

## :fallen_leaf:Concept

### Git workflow

- `HEAD`: TODO
- `working tree`: 工作区
- `index`: a **staging area** between your working directory and your repository.
- `repository`: or repo

<p align="center">
  <img
  src="https://i.loli.net/2020/02/20/1VT9wlzJgIv6hoi.png" width="80%">
</p>

<p align="center">
  <img
  src="https://i.loli.net/2020/02/20/CzpSm5FBP3MYdtO.png"
  width="70%">
</p>

**_References:_**

- [backlog: Git workflow](https://backlog.com/git-tutorial/git-workflow/)

### Changelog

用来记录 repo 的相关迭代信息

**_Ref:_** [Github olivierlacan/keep-a-changelog](https://github.com/olivierlacan/keep-a-changelog/blob/master/CHANGELOG.md)

### Licenses

许可

**_References:_**

- [Github: phodal/licenses](https://github.com/phodal/licenses)

### WIP

On GitHub, pull requests are prefixed by [WIP] to indicate that the pull requestor：

- has not yet finished his work on the code (thus, work in progress), but
- looks for have some initial feedback (early-pull strategy)
- wants to use the continuous integration infrastructure of the project. For instance, TravisCI, CodeCov, and codacy.

**_References:_**

- [stackoverflow: GitHub: What is a “wip” branch?](https://stackoverflow.com/questions/15763059/github-what-is-a-wip-branch)

### gitignore

- :thumbsup:[Bitbucket: .gitignore](https://www.atlassian.com/git/tutorials/saving-changes/gitignore)

#### :triangular_flag_on_post:Ignore a folder without a specific file

```shell
# OK
/test/*
!/test/README.txt
# Failed
/test/
!/test/README.txt
```

#### Ignore filename contain parttern

```shell
**.vscode** # 忽略所有文件名中包含`.vscode`的文件
```

**_Ref:_** [stackoverflow: git ignore filenames which contain <pattern to ignore>](https://stackoverflow.com/a/32335752/4636081)

## :fallen_leaf:Generate Key

```shell
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

**_References:_**

- [Github Help: Generating a new SSH key and adding it to the ssh-agent](https://help.github.com/en/enterprise/2.18/user/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

## :fallen_leaf:git config

### username and email

- [Github Help: Setting your username in Git](https://help.github.com/en/github/using-git/setting-your-username-in-git)

**Print current username and email**

```shell
git config user.name
git config user.email
```

**_References:_**

- [Blog: How to show or change your Git username or email address](https://alvinalexander.com/git/git-show-change-username-email-address)

#### Set global user name and email

```shell
# username
git config --global user.name "<your name>"
# email
git config --global user.email "<your email>"
```

#### Set single repo user name and email

```shell
# username
git config user.name "<your name>"
# email
git config user.email "<your email>"
```

## :fallen_leaf:git clone

### Clone a specific tag

```shell
git clone <git ssh>
git checkout tags/<tag_name> -b <new_branch_name>
```

**_Ref:_** [stackoverflow: Download a specific tag with Git](https://stackoverflow.com/questions/791959/download-a-specific-tag-with-git)

## :fallen_leaf:git checkout

### Checkout into a specific tag

```shell
git checkout <tag_name>
```

**_References:_**

- [CSDN: git 切换到某个 tag](https://blog.csdn.net/DinnerHowe/article/details/79082769)

### Checkout new branch from a specific tag

```shell
git checkout -b <branch_name> <tag_name>
```

### Checkout new branch from remote

```shell
git checkout -b <local_branch_name> <remote_name>/<remote_branch_name>
git checkout -t <remote_name>/<branch_name>
```

**_Ref:_** [stackoverflow: How do I check out a remote Git branch?](https://stackoverflow.com/a/1783426/4636081)

### Rename local and remote branch name

```shell
# 1. Rename local branch
git branch -m new-name
# 1. If you are on a differnet branch
git branch -m old-name new-name
# 2. Delete the old-name remote branch and push the new-name local branch
git push origin :old-name new-name
# 3. Rest the upstream branch for the new-name local branch.
# Switch to the branch and then
git push origin -u new-name
```

**_References:_**

- [Blog: Rename a local and remote branch in git](https://multiplestates.wordpress.com/2015/02/05/rename-a-local-and-remote-branch-in-git/)

## :fallen_leaf:git reset

TODO

- hard:
- soft:
- mix:

```shell
git reset --hard commit_id
```

**_References:_**

- :thumbsup:[博客园: git reset soft,hard,mixed 之区别深解](https://www.cnblogs.com/kidsitcn/p/4513297.html)
- [CSDN: git reset 版本回退](https://blog.csdn.net/ezhchai/article/details/79387369)
- [stackoverflow: What's the difference between git reset --mixed, --soft, and --hard?](https://stackoverflow.com/questions/3528245/whats-the-difference-between-git-reset-mixed-soft-and-hard)
- [git 7.7 Git 工具 - 重置揭密](https://git-scm.com/book/zh/v2/Git-%E5%B7%A5%E5%85%B7-%E9%87%8D%E7%BD%AE%E6%8F%AD%E5%AF%86)

---

### Reset remote url

```shell
git remote set-url origin <new_url>
```

**_References:_**

- [stackoverflow: How to change the URI (URL) for a remote Git repository?](https://stackoverflow.com/a/2432799/4636081)

## :fallen_leaf:git rm

### Untrack tracked file

Assume you have commit your .vscode, and now you don't want to track it anymore.

```shell
git rm -r --cached .vscode
```

**_References:_**

- [stackoverflow: Untrack tracked file in Git — but only in specific branch?](https://stackoverflow.com/questions/9368394/untrack-tracked-file-in-git-but-only-in-specific-branch)

## :fallen_leaf:git rebase

- :thumbsup::thumbsup:[TOWER: Rebase 代替合并](https://www.git-tower.com/learn/git/ebook/cn/command-line/advanced-topics/rebase)
- :thumbsup::thumbsup::thumbsup:[stackoverflow: Git merge master into feature branch](https://stackoverflow.com/a/16956062/4636081)

### 理解

### 1. 命令的含义

```shell
git checkout barnch-A # you are on branch-a
git rebase branch-B # rebase A on top of B
```

这条命令的含义是： 将 `branch-A` 搬移到 `branch-B` 上，即这条命令改变的是`branch-A`，对`branch-A`进行了换基操作。

- `LOCAL`: `branch-B`
- `REMOTE`: `branch-A`

**_Ref:_** :thumbsup::thumbsup::thumbsup:[stackoverflow: git rebase, keeping track of 'local' and 'remote'](https://stackoverflow.com/a/3052118/4636081)

### 2. 切记

如果你想向公共的分支提交 commit，只能使用 merge 的方式。即：不要在公共的分支上执行`git rebase xxx`， **only rebase private branches**

公共分支：该分支会接受多人的 commit

因为在公共的分支上执行`git rebase xxx`会导致公共分支的 commit 记录发生变化，从而导致别人的公共分支和你的不一样，该问题会很严重

因此，在 Github 中如果你想给开源仓库的 master 提交代码，需要提 pull request，当 request 被同意时，Github 会采用 pull 的方式将你的提交 merge 到主分支上。

### 3. 同步上下游分支

更新合作分支到自己的开发分支时的操作：

```shell
git checkout self-branch # 你的开发分支
git rebase remote-branch
解决冲突 (配合meld)
git rebase --continue
```

1. 下游分支更新上游分支内容的时候使用 rebase
2. 上游分支合并下游分支内容的时候使用 merge
3. 更新当前分支的内容时一定要使用 --rebase 参数

**_Ref:_** :thumbsup:[知乎: GIT 使用 rebase 和 merge 的正确姿势](https://zhuanlan.zhihu.com/p/34197548)

### 4. 解决 rebase 冲突

1. git mergetool
2. git rebase --continute
3. 重复 1 和 2，直到没有冲突

**_Ref:_** [Blog: git rebase 冲突解决步骤](https://hmgle.github.io/git/2013/08/22/git-rebase.html)

## :fallen_leaf:git fetch

TODO: FETCH_HEAD 和 HEAD 的区别

```shell
git fetch <remote_name>
```

> @rivergold: 这条命令的含义是：从 remote 获取最新的 commit 并更新`FETCH_HEAD`

**_References:_**

- [CSDN: 详解 git fetch 与 git pull 的区别](https://blog.csdn.net/riddle1981/article/details/74938111)

After `git fetch`, you can run `git merge` to merge `FETCH_HEAD` into current branch.

### Fetch a specific tag

```shell
git fetch [remote] <tag_name>
```

E.g.

```shell
git fetch pytorch v1.3.0
```

**_References:_**

- [stackoverflow: Fetch a single tag from remote repository](https://stackoverflow.com/questions/45338495/fetch-a-single-tag-from-remote-repository)

## :fallen_leaf:git merge

```shell
git checkout branch-A # you are on branch-A
git merge branch-B # merge B into A
```

> @rivergold: 这条命令的含义是：将 `branch-B` `merge` 到当前所处的 `branch-A` 中

- `LOCAL`: `branch-A`
- `REMOTE`: `branch-B`

**_Ref:_** :thumbsup::thumbsup::thumbsup:[stackoverflow: git rebase, keeping track of 'local' and 'remote'](https://stackoverflow.com/a/3052118/4636081)

## :fallen_leaf:git pull

```shell
# you are branch-1
git pull
```

> @rivergold: 这条命令的含义是：从远端 fetch branch-1 并 merge 到当前本地的 branch-1 中

### Pull from a local branch

When finish Gitlab or Github merge request / pull request, because of Gitlab or Github use `git merge` with `--no-ff`, `HEAD` of `dev` will fall behind `HEAD` of `master`. You need to do followings to fast-forward `HEAD` of `dev` to `master`.

```shell
git checkout master
git pull
git checkout dev
git pull . master
```

- `.`: means from local not remote

- `git pull . master`: means pull from `local:master` into current `dev`

**_References:_**

- [stackoverflow: How to “pull” from a local branch into another one?](https://stackoverflow.com/questions/5613902/how-to-pull-from-a-local-branch-into-another-one)

- [Samuel Gruetter Blog Git: Fast-forwarding a branch without checking it out](https://samuelgruetter.net/blog/2018/08/31/git-ffwd-without-checkout/)

- [ariya.io: Fast-Forward Git Merge](https://ariya.io/2013/09/fast-forward-git-merge)

## :fallen_leaf:git push

### Delete remote branch

```shell
git push <remote_name> --delete <branch_name>
```

**_References:_**

- [medium: Delete a local and a remote GIT branch](https://koukia.ca/delete-a-local-and-a-remote-git-branch-61df0b10d323)

## :fallen_leaf:git clean

### Delete untracked files

```shell
git clean -nf
```

### Delete untracked files and directory

```shell
git clean -nfd
```

### Use `-n` simulate what will delete

```shell
git clean -nfd
```

**_References:_**

- [CSDN: 删除 git 库中 untracked files（未监控）的文件](https://blog.csdn.net/ronnyjiang/article/details/53507306)

## :fallen_leaf: git stash

### git stash with custom message

```shell
git stash serve "your message"
```

**_Ref:_** [stackoverflow: How to name and retrieve a stash by name in git?](https://stackoverflow.com/questions/11269256/how-to-name-and-retrieve-a-stash-by-name-in-git)

## :fallen_leaf: git remote

### Refresh remote branches

When you delete remote branch, and you want to make your local remote branch as same as remote.

```shell
git remote update <remote_name> --prune
```

**_Ref:_** [stackoverflow: When does Git refresh the list of remote branches?](https://stackoverflow.com/questions/36358265/when-does-git-refresh-the-list-of-remote-branches)

## :fallen_leaf: git tag

### List tags

```shell
git tag
```

**_Ref:_** [stackoverflow: How to list all Git tags?](https://stackoverflow.com/a/1064505/4636081)

### Create tag

```shell
git tag <tag_name>
```

#### Create annotated tag

```shell
git tag <tag_name> -a
```

#### Create tag on specific commit id

```shell
git tag <tag_name> <commit_id> -a
```

### Push tag to remote

```shell
git push origin <tag_name>
```

**_References_**

- [stackoverflow: Create a tag in a GitHub repository](https://stackoverflow.com/a/18223354/4636081)
- [stackoverflow: How to tag an older commit in Git?](https://stackoverflow.com/questions/4404172/how-to-tag-an-older-commit-in-git)

### Update local tags from remote

```shell
git featch --tags
```

**_Ref:_** [stackoverflow: If a git tag changes on remote, a git fetch does not update it locally. Is this a bug?](https://stackoverflow.com/a/8433371/4636081)

### Delete tag

#### Delete local tag

```shell
git tag -d <tag_name>
```

#### Delete remote tag

```shell
git push --delete origin <tag_name>
```

**_Ref:_** [devconnected: How To Delete Local and Remote Tags on Git](https://devconnected.com/how-to-delete-local-and-remote-tags-on-git/)

## :fallen_leaf: git submodule

TODO: update

- [git doc: 7.11 Git Tools - Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- :thumbsup::thumbsup:[Medium: Git Submodule Cheatsheet](https://medium.com/faun/git-submodule-cheatsheet-29a3bfe443c3)

### `git submodule sync`

Synchronizes submodules' remote URL configuration setting to the value specified in `.gitmodules`.

**理解:** 更新`.gitmodules`中的一些配置信息

**_References:_**

- [stackoverflow: git submodule update vs git submodule sync](https://stackoverflow.com/questions/45678862/git-submodule-update-vs-git-submodule-sync/45679261)
- [stackoverflow: git submodule sync command - what is it for?](https://stackoverflow.com/questions/33739376/git-submodule-sync-command-what-is-it-for/33739770)

### Problem & Solution

### [Error] `git submodule update --init --recursive` occur `error: Server does not allow request for unadvertised object xxxx`

**Error**

```shell
error: Server does not allow request for unadvertised object 7f523de651585fe25cade462efccca647dcc8d02
Fetched in submodule path 'third_party/sleef', but it did not contain 7f523de651585fe25cade462efccca647dcc8d02. Direct fetching of that commit failed.
```

**Solution**

Run `git submodule sync` first

```shell
git submodule sync
git submodule update --init --recursive
```

**_References:_**

- :thumbsup:[Github AppImage/AppImageKit error: Server does not allow request for unadvertised object #511](https://github.com/AppImage/AppImageKit/issues/511#issuecomment-341885789)
- [git doc: 7.11 Git 工具 - 子模块](https://git-scm.com/book/zh/v2/Git-%E5%B7%A5%E5%85%B7-%E5%AD%90%E6%A8%A1%E5%9D%97)

**理解:**

- main repo
- submodule

### Update submodule to the latest commit with origin

TODO: 确认最正确的使用

```shell
git submodule sync
git submodule update --init --recursive # 和submodule的repo的origin同步
# git submodule update --remote
# Update specific submodule
git submodule update --init --recursive <specific path to submodule>
```

**_References:_**

- [stackoverflow: Update Git submodule to latest commit on origin](https://stackoverflow.com/questions/5828324/update-git-submodule-to-latest-commit-on-origin)
- [stackoverflow: How to only update specific git submodules?](https://stackoverflow.com/questions/16728866/how-to-only-update-specific-git-submodules)

### Reset submodule

```shell
# In main repo
git reset --hard
# Update submodule
git submodule update --recursive
```

**_References:_**

- [stackoverflow: How to revert a Git Submodule pointer to the commit stored in the containing repository?](https://stackoverflow.com/a/43303392/4636081)

### Remove submodule

```shell
# Remove the submodule entry from .git/config
git submodule deinit -f path/to/submodule

# Remove the submodule directory from the superproject's .git/modules directory
rm -rf .git/modules/path/to/submodule

# Remove the entry in .gitmodules and remove the submodule directory located at path/to/submodule
git rm -f path/to/submodule
```

**_References:_**

- [stackoverflow: How do I remove a submodule?](https://stackoverflow.com/a/21211232/4636081)

### 经验总结（感觉不太对）

- 使用`git submodule update --recursive`代替`git submodule update`
- `git submodule update`会更新 submodule 到该仓库的最新的 commit，但是最细你的 commit 可能不支持现有的 main repo，所以可能需要把 submodule `git reset`到某个版本

## :fallen_leaf:tig

A very good terminal git tool.

### Build from Source

#### Install ncurses to support Chinese

**Ubuntu**

```shell
apt install install libncurses5-dev libncursesw5-dev
```

**CentOS**

```shell
yum install ncurses-devel
```

**_Ref:_** [How To Install ncurses Library on a Linux](https://www.cyberciti.biz/faq/linux-install-ncurses-library-headers-on-debian-ubuntu-centos-fedora/)

#### Build

```shell
make configure # Only run if you are building from the Git repository
# Ubuntu
./configure LDFLAGS=-L/usr/local/opt/ncurses/lib CPPFLAGS=-I/usr/local/opt/ncurses/include
# CentOS
./configure LDFLAGS=-L/usr/lib64 CPPFLAGS=-I/usr/include
make -j8
make install
```

Check if tig build with ncursesw or not

```shell
$ tig --version

tig version 2.5.1
ncursesw version 5.9.20130511
```

**_Ref:_** [tig: Installation using configure](https://github.com/jonas/tig/blob/master/INSTALL.adoc#installation-using-configure)

### Bacis Command

- In `view-status`: `e`: edit file
- Refresh: `R` or `F5`

#### Create new branch

Edit `~/.tigrc`, add followings:

```shell
bind branch N !@git branch %(prompt)
bind branch D !?@git branch -d %(branch)
```

In `view-refs`: `n`

**_References:_**

- [Github - jonas/tig: Create new branch command #88](https://github.com/jonas/tig/issues/88)

#### Checkout into branch

In `view-refs`: `C`

#### Edit file

In `view-status`: `e`

#### Show commit id in main view

```shell
:toggle id
```

**_Ref:_** [Github-jonas/tig: Feature Request: Ability to show short commit hash in main view #340](https://github.com/jonas/tig/issues/340#issuecomment-458072445)

### Config `vim` as default git editor

```shell
git config --global core.editor "vim"
```

**_References:_**

- [How do I make Git use the editor of my choice for commits?](https://stackoverflow.com/a/2596835/4636081)

### tigrc

Config example is in [Github jonas/tig: tigrc](https://github.com/jonas/tig/blob/master/tigrc)

My `~/.tigrc` is like followings

```shell
bind branch N !@git branch %(prompt)
bind branch D !?@git branch -d %(branch)
```

### Problem & Solution

#### When run `tig`, occur `tig: No revisions match the given arguments.`

When you run `tig` in an empty git, it will occur this error.

**Solution:**

```shell
tig status
```

**_References:_**

- [Blog: tig 使用筆記](http://good5dog5.github.io/2016/03/31/tig-note/)
- [Github jonas/tig tig: No revisions match the given arguments. #435](https://github.com/jonas/tig/issues/435)

## :fallen_leaf:Tricks

### Fork and keep update with raw repo

1. `git remote add <forked_upstream_name> <forked_upstream_url>`

2. `git checkout master`

3. `git fetch <forked_upstream_name>`

4. `git rebase <forked_upstream_name>/master`

E.g.

```shell
# After fork pytorch/pytorch
git remote add pytorch https://github.com/pytorch/pytorch.git
git checkout master
git fetch pytorch
git merge pytorch/master
```

**_References:_**

- [简书: 如何解决 github fork 之后 更新问题](https://www.jianshu.com/p/840ea273f25a)

### Good way of setting Github notifications

- :thumbsup: :thumbsup:[Github cssmagic/blog: 如何正确接收 GitHub 的消息邮件 #49](https://github.com/cssmagic/blog/issues/49)

### References for how to replay in PR

- [pytorch/pytorch: Move hardtanh activation to Aten(CPU, CUDA) #30152](https://github.com/pytorch/pytorch/pull/30152)

### Download a single file from Github

**_References:_**

- [stackoverflow: Download single files from GitHub](https://stackoverflow.com/questions/4604663/download-single-files-from-github)

### Merge

- [segmentfault: 理解 git 结构与简单操作（四）合并分支的方法与策略](https://segmentfault.com/a/1190000013753251)

### Download a single folder or directory from a Github repo

**_References:_**

- [stackoverflow: Download a single folder or directory from a GitHub repo](https://stackoverflow.com/questions/7106012/download-a-single-folder-or-directory-from-a-github-repo)

### Rebase vs Merge

rebase: `rebase <from> onto <to>`

You can get complete introduction from [Git: 6 Git 分支 - 分支的衍合][git 6 分支 - 分支的衍合].

And there is a [video][gitkraken rebase vide] tell your how to use rebase in Gitkraken.

[gitkraken rebase vide]: https://www.youtube.com/watch?v=nAMbLbgxriI
[git 6 分支 - 分支的衍合]: https://git-scm.com/book/zh-tw/v1/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E8%A1%8D%E5%90%88

**Important: 不要给公共的 branch 换基**

例如：绝对不能将 master 换基到别的分支上，但是可以将别的分支换基到 master 上。

**_References:_**

- [Bitbucket: Merging vs. Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)

### Code Review

**_References:_**

- [Blog: 基于 GitLab 的 Code Review 教程](https://ken.io/note/gitlab-code-review-tutorial)

### How to use Watch, Star and Fork

- Watch: 关注这个项目的动态
- Star: 对这个项目点赞
- Fork: 相当于你自己有了一份原项目的拷贝，当然这个拷贝只是针对当时的项目文件，如果后续原项目文件发生改变，你必须通过其他的方式去同步

Ref [简书: 如何用好 github 中的 watch、star、fork](https://www.jianshu.com/p/6c366b53ea41)

### Merge two repos

Assum you want to merge `repo1` into `repo2`

1. Add `repo1` into `repo2` remote

   ```bash
   git remote add <remote name> <repo1 ssh>
   ```

2. Checkout repo1 to local

   ```bash
   git branch -a
   git checkout -b repo1 <remote name>/<branch name>
   ```

3. Merge repo1 into repot-master

   ```bash
   git checkout master
   git merge repo1
   ```

   If occur error `fatal: refusing to merge unrelated histories`

   Run:

   ```bash
   git merge repo1 --allow-unrelated-histories
   ```

   **_References:_** [CSDN: 解决 Git 中 fatal: refusing to merge unrelated histories](https://blog.csdn.net/wd2014610/article/details/80854807)

**_References:_** [CSDN: 合并两个 git 仓库](https://blog.csdn.net/gouboft/article/details/8450696)

### Keep gitignore file when switch branch

If your file is tracked before, and you want to ignore it.

```bash
git rm --cached file_path
```

**_Ref:_** [VoidCC: GIT：切换分支时如何保留忽略的文件？](http://cn.voidcc.com/question/p-volmanrm-hr.html)

**_References:_**

- [简书: git rm 与 git rm --cached](https://www.jianshu.com/p/337aeafc2f40)

### Pull Request

**_References:_**

- :thumbsup:[CSDN: Github 使用之 Pull Request 的正确打开方式（如何在 GitHub 上贡献开源项目）](https://blog.csdn.net/yxys01/article/details/78316649)
- :thumbsup:[知乎: GitHub Pull Request 入门](https://zhuanlan.zhihu.com/p/51199833)

### List remote tags

```shell
git ls-remote --tag [remote_name]
# Example-1
git ls-remote --tag
# Example-2
git ls-remote --tag pytorch
```

**_References:_**

- [stackoverflow: How to see remote tags?](https://stackoverflow.com/questions/25984310/how-to-see-remote-tags)

### Create independent branch

**_References:_**

- [简书: 创建 Git 独立分支](https://www.jianshu.com/p/504f26d59bc8)
- :thumbsup:[博客园: Git 以分支的方式同时管理多个项目](https://www.cnblogs.com/huangtailang/p/4748075.html)

## :fallen_leaf:Problems & Solutions

### `origin/dev` and `origin/master` is not at the same header

When you change `dev` and want to merge `dev` into `master`, you will do:

1. On `dev`, make a commit
2. check out into `master`
3. merge `dev` into `master`
4. push `master` to `origin/master`

After these operations, you will find that content of `dev` and `master` are same, but the `dev` is behind one commit of `master`. Your need to run followings additionally:

1. Checkout to `dev`
2. Fast-forward `dev` into `master`
3. Push `dev` into `origin/dev`

## :fallen_leaf:Tools

### VSCode

Good plugins:

- Git History: show git history
- GitLens: git command

### Lepton

[lepton_home]: https://github.com/hackjutsu/Lepton

A good code snippet manager based on Github Gist.

**_References:_**

- [What is the best code-snippets manager?](https://www.slant.co/topics/7247/~code-snippets-manager#10)

### Meld

**_Refererences:_**

- [Linux 中国: 给中级 Meld 用户的有用技巧](https://linux.cn/article-8808-1.html)

#### Config

```bash
[user]
    email = hejing01@qiyi.com
    name = Jing He
[core]
    autocrlf = input
[diff]
    tool = meld
[difftool]
    prompt = false
[difftool "meld"]
    cmd = /usr/bin/meld "$LOCAL" "$REMOTE"
[merge]
    tool = meld
[mergetool "meld"]
    cmd = /usr/bin/meld "$REMOTE" "$MERGED" "$LOCAL" --output "$MERGED"
```

**Note: I like current file into right hand window.**

Ref [stackoverflow: Setting up and using Meld as your git difftool and mergetool](https://stackoverflow.com/a/34119867/4636081)

#### Run `git difftool`

`$LOCAL`: pre-image (旧的)
`$REMOTE`: post-image (新的)

Ref [git docs](https://git-scm.com/docs/git-config#Documentation/git-config.txt-difftoollttoolgtcmd)

#### Run `git mergetool`

Suppose current in `branch-a`, merge `branch-b` into `a`.

`b ($REMOTE) -> a($LOCAL)`

#### [Problem] meld not work on Python 3.7

Edit `~/.gitconfig`:

```shell
[diff]
    tool = meld
[difftool]
    prompt = false
[difftool "meld"]
    cmd = "alias python=/usr/bin/python && alias python3=/usr/bin/python3.6 && /home/  rivergold/software/common/meld-3.20.1/bin/meld" "$LOCAL" "$REMOTE"
[merge]
    tool = meld
[mergetool "meld"]
    cmd = "alias python=/usr/bin/python && alias python3=/usr/bin/python3.6 && /home/  rivergold/software/common/meld-3.20.1/bin/meld"  "$REMOTE"  "$MERGED" "$LOCAL" --      output "$MERGED"
```

### gitignore.io

- [gitignore.io](https://gitignore.io/)

Many gitignore template.

**_Ref:_** [码农有道: 用好这几个工具，能大幅提升你的 Git/GitHub 操作效率！](https://mp.weixin.qq.com/s?__biz=MzIwNTc4NTEwOQ==&mid=2247487007&idx=1&sn=8b036cef813799bd9fa3728cfb4b6552&chksm=972adf65a05d567330f5f6480c9a3f6466a45898b4d5dad41e0a0f2259ccc6c8ae3330ced3f2&mpshare=1&scene=1&srcid=#rd)

### readme-md-generator

- [Github](https://github.com/kefranabg/readme-md-generator)