# Git

## GitKraken

[Official doc](https://support.gitkraken.com/start-here/interface)

<!--  -->
<br>

***
<!--  -->

## Concept

### Git workflow

- `working tree`:
- `index`: a **staging area** between your working directory and your repository.
- `repository`: or repo
    <p align="center">
      <img
      src="https://rivergold-images-1258274680.cos.ap-chengdu.myqcloud.com/git-index.png?q-sign-algorithm=sha1&q-ak=AKIDBIevgMFRj61KAvJH0wrMbHKFiYL0Fi8S&q-sign-time=1544529129;1544530929&q-key-time=1544529129;1544530929&q-header-list=&q-url-param-list=&q-signature=8c7e3e234c04469478df6340dfe0ed125535d418&x-cos-security-token=1e92f604019d0d93df5cb871a1a3443eb030703510001" width="90%">
    </p>

***References:***

- [backlog: Git workflow](https://backlog.com/git-tutorial/git-workflow/)

## Common command

### Add submodule

```bash
git submodule add <https> <path>
```

***References:***

- [简书: Git submodule 子模块的管理和使用](https://www.jianshu.com/p/9000cd49822c)


### `reset` `soft` `hard` `mixed`

***References:***
- [Blog: git reset soft,hard,mixed之区别深解](https://www.cnblogs.com/kidsitcn/p/4513297.html)

<!--  -->
<br>

***
<!--  -->

## Others

- [http://ohshitgit.com/](http://ohshitgit.com/)

## Branch

- `git checkout -b <branch_name>`: New a branch
- `git push <remote_name> <local branch_name>:<remote_branch_name>`: Push branch into **remote**

## From remote to local

- `git fetch <remote_name> <remote_branch_name>`: Fetch code from remote but it will not change your local code

- `git pull <remote_name> <remote_branch_name>:<local_branch_name>`: Pull a branch of remote and merge with local branch.

- `git pull <remote_name> <remote_branch_name>`: Pull a branch of remote and merge with local master

## From local to remote

- `git push <remote_name> <local_branch_name>:<remote_branch_name>`

***Referneces:***

- [git](https://git-scm.com/book/zh/v1/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E6%96%B0%E5%BB%BA%E4%B8%8E%E5%90%88%E5%B9%B6)
- [Blog: git常用命令](http://www.cnblogs.com/springbarley/archive/2012/11/03/2752984.html)

# Tricks

## Download a single file from Github

***References:***

- [stackoverflow: Download single files from GitHub](https://stackoverflow.com/questions/4604663/download-single-files-from-github)

## Download a single folder or directory from a Github repo

***References:***

- [stackoverflow: Download a single folder or directory from a GitHub repo](https://stackoverflow.com/questions/7106012/download-a-single-folder-or-directory-from-a-github-repo)