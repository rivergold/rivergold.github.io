---
title: "VSCode Configuration"
last_modified_at: 2020-02-22
categories:
  - Memo
tags:
  - Tool
  - VSCode
---

A collection about how to config VSCode for better develop environment.

## :fallen_leaf:Basics

- [VS Code 快捷键（中英文对照版）](https://segmentfault.com/a/1190000007688656)

### Open File in New Window

- On windows/Linux, press `Ctrl` + `k`, then release the keys and press `o`
- On macOS, press `CMD` + `k`, then `o`

This will open the active file tab in a new window/instace.

**_Ref_** [stackoverflow: Visual Studio Code open tab in new window](https://stackoverflow.com/questions/43362133/visual-studio-code-open-tab-in-new-window)

### Preview Mode

When in **preview mode**, file name is italic.

- `preview mode`: single-click file in sidebar
- `edit mode`: double-click file in sidebar or single-click it in the sidebar then double click the title

**_Ref:_** [stackoverflow: Open files always in a new tab](https://stackoverflow.com/questions/38713405/open-files-always-in-a-new-tab)

## :fallen_leaf:Code format

- Mac: `Shift` + `Option` + `F`
- Linux: `Ctrl` + `Shift` + `i`
- Windows: `Shift` + `Alt` + `F`

**_Ref_** [stackoverflow: How do you format code in Visual Studio Code (VSCode)](https://stackoverflow.com/questions/29973357/how-do-you-format-code-in-visual-studio-code-vscode)

My `User settings` about format:

```json
// -----Formatting
"editor.formatOnSave": true,
// Cpp
"[cpp]": {
  "editor.defaultFormatter": "xaver.clang-format",
  "editor.tabSize": 2,
},
"C_Cpp.clang_format_style": "Google",
// Python
"python.formatting.provider": "yapf",
// -----
```

### Set Insert New Line at The End of File

`Settings` -> `insert final newline`

**_Ref_** [stackoverflow: Visual Studio Code — Insert New Line at the End of Files](https://stackoverflow.com/questions/44704968/visual-studio-code-insert-new-line-at-the-end-of-files)

### Python Format

Using **yapf**, I think it's better than pep8.

**_References:_**

- [Google Python Style Guide: Background](http://google.github.io/styleguide/pyguide.html#1-background)
- [VSCode: Edit Python in Visual Studio Code](https://code.visualstudio.com/docs/python/editing#_formatterspecific-settings)

#### Position of Brace `{}` and Parenthesis `()`

```python
# Aligned with opening delimiter
foo = long_function_name(var_one, var_two,
                         var_three, var_four)
meal = (spam,
        beans)
# Aligned with opening delimiter in a dictionary
foo = {
    long_dictionary_key: value1 +
                         value2,
    ...
}
# 4-space hanging indent; nothing on first line
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)
meal = (
    spam,
    beans)
# 4-space hanging indent in a dictionary
foo = {
    long_dictionary_key:
        long_dictionary_value,
    ...
}
```

**注：** 圆括号需要追加在末尾，花口号可以在新的一行。

**_Ref_** [Google Python Style Guide: 3.4 Indentation](http://google.github.io/styleguide/pyguide.html#34-indentation)

### C++

Using `clang-format` and set Google style.

`clang-format` use `.clang-fromat` file to config

**Google style `.clang-format`**

```shell
# Run manually to reformat a file:
# clang-format -i --style=file <file>
BasedOnStyle: Google
DerivePointerAlignment: false
```

**_Ref_** [CSDN: Clang-Format 格式化选项介绍](https://blog.csdn.net/softimite_zifeng/article/details/78357898)

#### Change Tab Size into 2

In C++ file, indent size is 2. You need to set: `editor.detectIndentation` to `false`. And set:

```json
"[cpp]":{
  "editor.tabsize": 2,
}
```

**_Ref_** [stackoverflow: How to change indentation in Visual Studio Code?](https://stackoverflow.com/a/45671704/4636081)

## :fallen_leaf:Linting

### Python

Use `pylint` for Google python style.

```json
"python.linting.pylintEnabled": true
```

**_Ref_** [Google Python Style Guide: 2 Python Language Rules](http://google.github.io/styleguide/pyguide.html#21-lint)

### C++

Use vscode C/C++ extension.

#### Config VSCode Use C++11

settings -> C++ -> choose cpp standard

**_References:_**

- [VSCode doc: Configure the compiler path](https://code.visualstudio.com/docs/cpp/config-clang-mac#_configure-the-compiler-path)
- [Github Gist BaReinhard/VSCodeCPPSetup.md: Setting Up VS Code for C++ 11](https://gist.github.com/BaReinhard/c8dc7feb8b0882d13a0cac9ab0319547)

## :fallen_leaf:Debug

### Change Python environments in VS Code

**_Ref_** [Visual Studio Code Doc: Using Python environments in VS Code](https://code.visualstudio.com/docs/python/environments)

### Python Debugger

**_Ref_** [Blog: 用 VScode 代码调试 Python](https://www.cnblogs.com/it-tsz/p/9022456.html)

## :fallen_leaf:Other Settings

<!-- ## Solve `pylint` cannot recognize `cv2`

```json
// Pylint
"python.linting.pylintArgs": [
    "--extension-pkg-whitelist=cv2"
],
```

Ref [stackoverflow: How do I get PyLint to recognize numpy members?](https://stackoverflow.com/questions/20553551/how-do-i-get-pylint-to-recognize-numpy-members)

**_References:_**

- [stackoverflow: PyLint not recognizing cv2 members](https://stackoverflow.com/a/51916065/4636081) -->

### VSCode Ignore(exclude) Some Folder or Files

1. `File` -> `Preferences` -> `Settings`
2. Pick `workspace settings`
3. Add files or folders you want to exclude into `file.exclude`

Ref [stackoverflow: How can I exclude a directory from Visual Studio Code “Explore” tab?](https://stackoverflow.com/questions/33258543/how-can-i-exclude-a-directory-from-visual-studio-code-explore-tab)

i.e.

```json
{
  "files.exclude": {
    "dataset/": true,
    "**/__pycache__/": true
  }
}
```

Note: `**` represent find partten recursive. `**/__pycache__` will match all folder named `__pycache__`.

Ref [VSCode doc: Advanced search options](https://code.visualstudio.com/docs/editor/codebasics#_advanced-search-options)

**_Ref_** [Github microsoft/vscode: .pyc exclude not working #20458](https://github.com/Microsoft/vscode/issues/20458)

### Config Different Index Size for Different File Format

You need do two steps:

1. Uncheck `Editor: Detect Indentation` in User Settings

2. Edit `settings.json`

   ```json
   "editor.tabSize": 4,
   "editordetectIndentation":false,
   "[markdown]": {
       "editor.tabSize": 2
   },
   "[C++]":{
       "editor.tabSize": 2
   },
   ```

**_References:_**

- [stackoverflow: How to customize the tab-to-space conversion factor when using Visual Studio Code?](https://stackoverflow.com/a/43883133/4636081)
- [Visual Studio Code Doc: User and Workspace Settings](https://code.visualstudio.com/docs/getstarted/settings#_settings-file-locations)

### :thumbsup:Solve Python Autocomplete Not Work for `cv2`

My `opencv-python` is build from source, because I want to use FFMPEG with `x264`. The opencv-python install from `pip install opencv-python`'s FFMPEG is not with `x264`.

**Best solution of opencv-python with FFMPEG**

Follow this way, you don't need to build opencv-python from source.

1. pip install opencv-python
2. pip uninstall ffmpeg

VSCode intellisense fails for `cv2`, the reason is that VSCode autocomplete cannot find where is `cv2`. You need to add the `cv2` path into `"python.autoComplete.extraPaths"`

The `cv2` directory is

```shell
cv2
├── config-3.7.py
├── config.py
├── __init__.py
├── load_config_py2.py
├── load_config_py3.py
├── __pycache__
│   ├── __init__.cpython-37.pyc
│   └── load_config_py3.cpython-37.pyc
└── python-3.7
    └── cv2.cpython-37m-x86_64-linux-gnu.so
```

Edit your VSCode `settings.json`

```json
"python.autoComplete.extraPaths": [
    "<anaconda site-packages path>/cv2/python-3.7",
],
```

### Solve VSCode PyLint Doesn't Recognized `cv2`

Edit your VSCode `settings.json`

```json
"python.linting.pylintArgs": ["--generate-members=cv2.*"],
```

**_References:_**

- :thumbsup:[Github PyCQA/pylint: "Module 'numpy' has no ... member" error even with extension-pkg-whitelist=numpy set in pylint.rc #779](https://github.com/PyCQA/pylint/issues/779#issuecomment-533455857)
- [Github PyCQA/pylint: cv2 module members are not recognized #2426](https://github.com/PyCQA/pylint/issues/2426#issuecomment-541537420)

### Solve VSCode PyLint Doesn't Recognized `torch`

Edit your VSCode `settings.json`

```json
"python.linting.pylintArgs": [
"--generated-members=torch.* ,cv2.* , cv.*"
]
```

**_Ref_** :thumbsup::thumbsup::thumbsup:[Github pytorch/pytorch: [Minor Bug] Pylint E1101 Module 'torch' has no 'from_numpy' member #701](https://github.com/pytorch/pytorch/issues/701#issuecomment-438215838)

### Remove Trailing Spaces, Same With Vim

`Settings` -> `Trim Trailing Whitespace`

**_Ref_** [stackoverflow: Remove trailing spaces automatically or with a shortcut](https://stackoverflow.com/questions/30884131/remove-trailing-spaces-automatically-or-with-a-shortcut)

### Trim Trailing Whitespace

```json
"files.trimTrailingWhitespace": true
```

> @rivergold: 设置该参数，vscode 会默认将悬挂的空格转化为空行。如果你设置了 tab 为 n 个空格时，vscode 也会将其转化为空行

**_Ref_** [CSDN: VS code 保存文件后自动删除多余空格](https://blog.csdn.net/Crazy_Sakura/article/details/88707414)

## :fallen_leaf:Extentions

**_Ref_** [知乎: 强大的 VS Code 入门](https://zhuanlan.zhihu.com/p/64779441?utm_source=ZHShareTargetIDMore&utm_medium=social&utm_oi=37839882420224)

### Setting Sync

Sync your VSCode settings via Gist.

#### Shortcuts

**Linux**

- Upload settings: `shift` + `alt` + `u`

- Download settings: `shift` + `alt` + `D`

**macOS**

- Upload settings: `shift` + `option` + `u`

- Download settings: `shift` + `option` + `d`

**_Ref_** [Settings Sync](https://marketplace.visualstudio.com/items?itemName=Shan.code-settings-sync)

### TODO-highlight & TODO Tree

#### Set color

Edit `settings.json`

```bash
// -----todo-highlight
"todohighlightkeywords": [{
    "text": "TODO:",
    "backgroundColor":"rgba(0, 0, 128,0.9)"
}]
// -----
```

**_Ref_** [Github wayou/vscode-todo-highlight: How to customize built-in colors? #18](https://github.com/wayou/vscode-todo-highlight/issues/18)

### Bookmarks

Set bookmark in code file.

### Code Spell Checker

Check if there has any spell mistake.

### C++

Basic extension for C++.

### C/C++ Clang Command Adapter

Support clang.

**_Ref:_** [知乎: Visual Studio Code 如何编写运行 C、C++ 程序？](https://www.zhihu.com/question/30315894)

## :fallen_leaf:Remote Edit Extension

### sftp

- :thumbsup:[liximomo/vscode-sftp](https://github.com/liximomo/vscode-sftp)

<!-- - [humy2833/FTP-Simple](https://github.com/humy2833/FTP-Simple) -->

**vscode-sftp**

Press `F1` to config, here is an example:

```json
{
  "name": "name",
  "host": "remote ip",
  "protocol": "sftp",
  "port": <remote sftp port>,
  "username": "username",
  "remotePath": "/root/rivergold-project/pytorch",
  "uploadOnSave": true,
  "privateKeyPath": "/home/rivergold/.ssh/id_rsa",
  "ignore": [".vscode", ".git", ".DS_Store", ".pyc", ".so"],
  "ignoreFile": ".gitignore"
}
```

**_Ref_** [Github liximomo/vscode-sftp: possible to support reading gitignore file as for ignoring file #142](https://github.com/liximomo/vscode-sftp/issues/142)

#### [Error] When in sftp remote explorer and edit file, occur `Cannot edit in read only file`

**Solution**

VScode > File > Preferences > Settings

Select "downloadWhenOpenInRemoteExplorer" then restart vscode.

**_Ref_** :thumbsup:[Github liximomo/vscode-sftp: Files are read-only when opening from sftp remote explorer #270](https://github.com/liximomo/vscode-sftp/issues/270#issuecomment-501946457)

### :bulb::thumbsup:Code Server

Use VSCode in web server, recommended use this tool!

[Github](https://github.com/cdr/code-server)

#### Basic Use

```shell
export PASSWORD="<>your_password"
code-server --host 0.0.0.0 --port 8080
```

**_Ref_** [Github cdr/code-server: How to change the password #548](https://github.com/cdr/code-server/issues/548)

#### Install Extension

```shell
code-server --install-extension <extension_name>
```

**_Ref_** [Github cdr/code-server: Installing extensions from command line #171](https://github.com/cdr/code-server/issues/171#issuecomment-557789729)

#### Common extension

Some extension is differnt from raw VSCode extension

- `ms-python.python`

#### Problem & Solution

**_Ref:_** [Github microsoft/vscode-remote-release: Cannot connect to remote machine over ssh - no way to debug because the terminal is not kept open #243](https://github.com/microsoft/vscode-remote-release/issues/243#issuecomment-490765625)

## :fallen_leaf:基于 vscode 开发环境搭建

### 基本插件

Updating

- Bookmarks
- Bracket Pair Colorizer
- Code Spell Checker
- GitLens
- Git History
- Material Icon Theme
- Path Intellisense
- Project Manager
- prototxt
- Remote Development
- Settings Sync
- SFTP
- shell-format
- TODO Highlight
- Todo Tree
- vscode-fileheader
- vscode-icons
- vscode-json

### Settings

```json
// Font size
"editor.fontSize": 14,
// Teminal font size
"terminal.integrated.fontSize": 15,
// Tab size
"editor.tabSize": 4,
// Close mouse middle button paste
"editor.selectionClipboard": false,
// Close detect indentation
"editor.detectIndentation": false,
// Auto remove trailing whitespace
"files.trimTrailingWhitespace": true,
// Auto format when saving
"editor.formatOnSave": true,
// Insert final newline
"files.insertFinalNewline": true,
// -----todo-highlight
"todohighlight.keywords": [
    {
        "text": "TODO:",
        "backgroundColor": "rgba(100,149,237,0.9)"
    }
],
```

### C++ 开发环境搭建

- vscode plugin: C/C++
- vscode plugin: clang-format, edit `.clang-format` to use Google style
- vscode plugin: CMake
- vscode plugin: vscode-cudacpp
- cmake
- vscode plugin: cmake-format, need `pip install cmake-format`
- ninja
- ccache

#### vscode 配置

```json
"[cpp]": {
    "editor.defaultFormatter": "xaver.clang-format",
    "editor.tabSize": 2
},
"C_Cpp.clang_format_style": "Google",
```

**Config vscode use c++11**

settings -> C++ -> choose cpp standard

#### `.clang-format`配置

```shell
# Run manually to reformat a file:
# clang-format -i --style=file <file>
BasedOnStyle: Google
DerivePointerAlignment: false
```

**_Ref_** [Github tensorflow/tensorflow: .clang-format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/.clang-format)

#### `.vscode/c_cpp_properties.json`

**_Ref_** :thumbsup:[VSCode doc: c_cpp_properties.json reference](https://code.visualstudio.com/docs/cpp/c-cpp-properties-schema-reference)

### Python 开发环境搭建

- vscode plugin: Python
- vscode plugin: autoDocstring
- Anaconda
- pylint
- yapf

#### vscode 配置

```json
"python.linting.enabled": true,
"python.linting.pylintArgs": [
    "--generated-members=cv2.*"
],
// "python.linting.pylintPath": "/home/rivergold/software/anaconda/bin/pylint",
"python.formatting.provider": "yapf",
```

### Makrdown 环境搭建

- vscode plugin: Markdown Preview Enhanced
- vscode plugin: markdownlint

#### vscode 配置

```json
// Markdownlint
"markdownlint.config": {
    "MD025": false,
    "MD033": false,
    "MD007": {
        "indent": 4
    }
},
```

#### Markdown to PDF

- :thumbsup:[markdown-preview-enhanced doc: Pandoc](https://shd101wyy.github.io/markdown-preview-enhanced/#/pandoc)

**Note: Pandoc convert result is not very good for some markdown grammar.**

### Blog 环境搭建

- :thumbsup:[Blog: 纯文本做笔记 --- 使用 Pandoc 与 Markdown 生成 PDF 文件](https://jdhao.github.io/2017/12/10/pandoc-markdown-with-chinese/)

#### Dependency

- Pandoc
- Tex Live: Pandoc use LaText to convert markdown into pdf

#### Install

##### 1. Pandoc

markdownlint need Pandoc >= 2.0

Download latest Pandoc from [Github](https://pandoc.org/installing.html) and install.

##### 2. TexLive

**Ubuntu**

```shell
sudo apt install textlive-latex-base
sudo apt install texlive-fonts-recommended
sudo apt install texlive-xetex
```

**_References:_**

- [LinuxConfig: How to install LaTex on Ubuntu 18.04 Bionic Beaver Linux](https://linuxconfig.org/how-to-install-latex-on-ubuntu-18-04-bionic-beaver-linux)

- [Github Gist: rain1024/tut.md](https://gist.github.com/rain1024/98dd5e2c6c8c28f9ea9d)

- [stackoverflow: Is it possible to install only the required LaTeX tool “pdflatex” in Ubuntu?](https://tex.stackexchange.com/questions/112734/is-it-possible-to-install-only-the-required-latex-tool-pdflatex-in-ubuntu)

#### Problem

##### [Error] Pandoc `pdflatex not found. pdflatex is needed for pdf output`

You need to install pdflatex or TexLive (this contain pdflatex)

##### [Error] Pandoc `Error producing PDF. ! Font T1/cmr/m/n/10=ecrm1000 at 10.0pt not loadable`

```shell
sudo apt install texlive-fonts-recommended
```

**_Ref_** [Github jgm/pandoc: Which packages do I need on Ubuntu? #1894](https://github.com/jgm/pandoc/issues/1894)

##### [Error] Pandoc `Error: Unicode char 视 (U89C6) (inputenc) not set up for use with LaTeX.`

- Use `-V CJKmainfont="<Chinese Font>"` in Pandoc command
- use `--pdf-engine=xelatex` instead of `pdflatext`