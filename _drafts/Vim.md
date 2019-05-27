# Vim

# Build and Install

***If you want to use vim with anaconda Python well, strongly suggest to build vim from source.***

# Config

Strongly suggest to use **[The Ultimate vimrc](https://github.com/amix/vimrc)**.

**Note:** You maybe say that SpaceVim is good, but I think it is beautiful but is slow when using.

## Install `The Ultimate vimrc`

```bash
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh
```

## Custom config

### Config vim

Just edit vim `~/.vim_runtime/my_configs.vim`

### Instal plugin

**Note:** **The Ultimate vimrc** install vim pathogen.

```bash
cd ~/.vim_runtime
git clone <the plugin> # Follow the plugin installation
```

### My `my_configs.vim` and plugins

#### Plugins

- gruvbox

    ```bash
    git clone https://github.com/morhetz/gruvbox.git ~/.vim_runtime/
    ```

    Edit `my_configs.vim`

    ```vim
    colorscheme gruvbox
    ```

    ***Ref:*** [Github morhetz/gruvbox: Installation](https://github.com/morhetz/gruvbox/wiki/Installation)

## Common Use

The leader is `,`

### NERD Tree

```bash
map <leader>nn :NERDTreeToggle<cr>
map <leader>nb :NERDTreeFromBookmark 
map <leader>nf :NERDTreeFind<cr>
```


# Problems & Solutions

## YouCompleteMe

### When run `vim` occur `The ycmd server SHUT DOWN (restart with ':YcmRestartServer'). Unexpected error while loading the YCM core library. Type ":YcmToggleLogs ycmd_58963_stderr_d1zdkhog.log" to check the logs.`

After run `:YcmToggleLogs ycmd_58963_stderr_d1zdkhog.log`, occur followings:

```bash
 Traceback (most recent call last):
   File "/home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycmd/utils.py", line 637, in ImportAndCheckCore
     ycm_core = ImportCore()
   File "/home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycmd/utils.py", line 628, in ImportCore
     import ycm_core as ycm_core
 ImportError: /home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycm_core.so: undefined symbol:                    clang_getCompletionFixIt
```

Maybe your llvm version is too low(need >= 7.0.0).

- [ ] How to check llvm version ?

***Ref:*** [Github Valloric/YouCompleteMe: When I enter VIM I get The ycmd server SHUT DOWN (restart with ':YcmRestartServer'). Unexpected error while loading the YCM core library. Type ':YcmToggleLogs ycmd_49739_stderr_rbhx3wqr.log' to check the logs #3236](https://github.com/Valloric/YouCompleteMe/issues/3236#issuecomment-439987788)

***References:***

- [博客园: 发现vi出现此错误~/.vim/bundle/YouCompleteMe/third_party/ycmd/ycm_core.so: undefined symbol: clang_getCompletionFixIt](https://www.cnblogs.com/dakewei/p/10491485.html)

### When build YouCompleteMe occur error `make[3]: *** No rule to make target '/home/rivergold/software/anaconda/lib/python3.7/config-3.7m-x86_64-linux-gnu/libpython3.7m.so', needed by '/home/rivergold/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/ycm_core.so'.  Stop.`

When you build YouCompleteMe with Anaconda Python, maybe occur this error.

***Ref:*** [CSDN: YouCompleteMe+anaconda+vim8.0自动补全](https://blog.csdn.net/u013806541/article/details/72057272)

- [ ] The solution is not good.

<!-- # SpaceVim

## Install

```bash
curl -sLf https://spacevim.org/install.sh | bash
```

***Ref:*** [SpaceVim doc: Quick start guide](https://spacevim.org/quick-start-guide/)

## Uninstall

```bash
curl -sLf https://spacevim.org/install.sh | bash -s -- --uninstall
```

***Ref:*** [Github SpaceVim/SpaceVim: 如何卸载 how to uninstall #941](https://github.com/SpaceVim/SpaceVim/issues/941#issuecomment-445451297)

## Config



Edit ``

```
relativenumber = false # Not use relative line number
``` -->