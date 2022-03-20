---
title: 使用Flask搭建文件服务器
categories: Tech
tags:
  - Python
  - Flask
date: 2021-02-02 21:42:28
---


远程开发时经常需要查看服务器上的图片、视频，有一个支持下载和上传的文件服务器还是很方便的。搭建文件服务器的方式有很多，例如基于 Apache，Nginx 等，本文介绍使用 Flask 和 Python 构建自己的小型文件服务器，方便日常开发使用。

作为一个算法工程师，前端、后端开发确实不是强项，实现的都是基础功能，如有不足之处还望见谅。

## 目录功能

关于如何在 Flask 上快捷的实现文件目录功能，我调研了很久。终于在 stackoverflow 上发现了一个不错的 Flask 插件[Flask-AutoIndex](https://github.com/general03/flask-autoindex)，即插即用，nice！

### 安装 Flask-AutoIndex

```shell
pip install Flask-AutoIndex
```

### 为 app 或者 blueprint 添加 autoindex 功能

Flask-AutoIndex 的使用很简单，主要代码如下

```python
import os.path
from flask import Flask
from flask_autoindex import AutoIndex

app = Flask(__name__)
AutoIndex(app, browse_root=os.path.curdir)

if __name__ == '__main__':
    app.run()
```

`AutoIndex`可以传入 app 构建，也可以传入 blueprint 构建，其 url 为默认的`/`，目前还没有参数接口进行用户自定义。

Index 的界面如下：

<p align="center">
  <img
  src="https://i.loli.net/2021/02/02/wtqCkKGzcFbBi45.png" width="80%">
</p>

## 上传功能

### html 实现

```html
<!doctype html>
<title>Upload new File</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
```

`name=file`会在后面的视图函数中使用。

### 视图函数实现

代码中的注释说明了各个核心步骤：

```python
import os
from flask import render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from . import home_bp


@home_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # 检查request是否有name=file的字段
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # 检查file的文件名是否合规
        if file.filename == '':
            flash('No selected file')

        # 基于werkzeug.utils.secure_filename获取文件之后再保存
        # 之后使用url_for获取视图函数uploaded_file的url，之后使用redirect返回该url的页面
        if file:
            filename = secure_filename(file.filename)
            file_save_path = home_bp.config['upload_dir'] / filename # 将文件保存在upload_dir路径中
            file.save(file_save_path.as_posix())
            return redirect(url_for('home.uploaded_file', filename=filename))

    return render_template('upload.html')


@home_bp.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(home_bp.config['upload_dir'], filename)
```
