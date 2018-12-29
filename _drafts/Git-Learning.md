# Concept

### Git workflow

- `working tree`:
- `index`: a **staging area** between your working directory and your repository.
- `repository`: or repo
    <br>
    <p align="center">
      <img
      src="https://rivergold-images-1258274680.cos.ap-chengdu.myqcloud.com/git-index.png?q-sign-algorithm=sha1&q-ak=AKIDLyeuzvSPVPTk6b5N8nLeI9vNhzL5y4XD&q-sign-time=1544779993;1544781793&q-key-time=1544779993;1544781793&q-header-list=&q-url-param-list=&q-signature=c5f889b2799233522d5437849c9517b463a3f4ab&x-cos-security-token=e52cdfff8d78dca2e4b3d8c6c60bc8ac1f4771b510001" width="90%">
    </p>

***References:***

- [backlog: Git workflow](https://backlog.com/git-tutorial/git-workflow/)

<br>

***

<br>

# Basic

## Local

### `git status`

- `Untracked files`: New add file and not be untracked

- `Changes to be committed`: After use `git add <file>`, the file is tracked and staged

- `Changes not staged for commit`: Tracked file is changed, but not staged. Use `git add` to stage.

### `git add <file>`

Track new file or modified file, and stage file.

### `git commit -m <message>`

Commit staged file into `Repository`.

### `git submodule add <git https/ssh> <folder name>`

***References:***

- [简书: Git submodule 子模块的管理和使用](https://www.jianshu.com/p/9000cd49822c)

### `git reset [--hard|--soft|--mix]`

***References:***

- [stackoverflow: What's the difference between git reset --mixed, --soft, and --hard?](https://stackoverflow.com/questions/3528245/whats-the-difference-between-git-reset-mixed-soft-and-hard)
- [git 7.7 Git 工具 - 重置揭密](https://git-scm.com/book/zh/v2/Git-%E5%B7%A5%E5%85%B7-%E9%87%8D%E7%BD%AE%E6%8F%AD%E5%AF%86)

<br>

***

<br>

## Remote

### From remote

#### `git fetch [remote-name]`

Only Get files from remote to your repository. It will not change anythings.

#### `git pull`

Get files from remote and merge into current branch.

#### `git clone --recurese-submodules <git https/ssh>`

Clone with submodule from remote.

For already cloned repos, or older Git versions, use:

```bash
git clone git://github.com/foo/bar.git
cd bar
git submodule update --init --recursive
```

***References:***

- [stackoverflow: How to git clone including submodules?](https://stackoverflow.com/questions/3796927/how-to-git-clone-including-submodules)

***

### To remote