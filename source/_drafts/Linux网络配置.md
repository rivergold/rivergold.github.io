---
title: Linux网络配置
categories: Tech
date: 2022-03-26 11:57:55
tags:
- Linux
---

日常开发基本都是在Linux上个进行的，有一个好的开发环境至关重要。rivergold整理一份关于Linux网络配置的笔记，主要包括以下内容：

- 防火墙配置
- 设置静态IP
- 代理配置
- 使用Flask搭建简易的文件服务器
- NFS远程挂载

## Linux防火墙命令

不同的Linux发行版有各自不同的防火墙命令，这里主要介绍两种：

- Ubuntu：`ufw`
- CentOS：`firewall-cmd`

### 安装

{% tabs 安装 %}
<!-- tab Ubuntu -->
```bash
sudo apt install ufw
```
<!-- endtab -->
<!-- tab CentOS -->
```bash
yum install firewalld firewall-config
```
<!-- endtab -->
{% endtabs %}

### 启动

{% tabs 启动 %}
<!-- tab Ubuntu -->
```bash
sudo ufw enable
```
<!-- endtab -->
<!-- tab CentOS -->
```bash
sudo systemctl enable firewalld.service
sudo systemctl start firewalld.service
```
<!-- endtab -->
{% endtabs %}

### 关闭

{% tabs 安装 %}
<!-- tab Ubuntu -->
```bash
sudo ufw disable
```
<!-- endtab -->
<!-- tab CentOS -->
```bash
sudo systemctl stop firewalld.service
sudo systemctl disable firewalld.service
```
<!-- endtab -->
{% endtabs %}

### 查看状态

{% tabs 查看状态 %}
<!-- tab Ubuntu -->
```bash
sudo ufw status
```
<!-- endtab -->
<!-- tab CentOS -->
```bash
sudo systemctl status firewalld
```
<!-- endtab -->
{% endtabs %}

### 开启端口

{% tabs 开启端口 %}
<!-- tab Ubuntu -->
基础使用

```bash
# sudo ufw allow <port>/<tcp/udp>
sudo ufw allow ssh # 允许所有外部ip访问ssh端口（默认22）
sudo ufw allow 22/tcp # 允许所有外部ip访问本机的22/tcp
sudo ufw allow 63200:63205/tcp # 允许所有外部ip访问本机的63200到63205/tcp
```

限制IP

```bash
# sudo ufw allow from <from_ip> to any port <port>
sudo ufw allow from 192.168.1.5 # 允许192.168.1.5访问本机所有端口
sudo ufw allow from 192.168.1.5 to any port 80 # 允许192.168.1.5访问本机80端口
sudo ufw allow from 192.168.1.5 to any port 80 proto tcp # 允许192.168.1.5访问本机80/tcp端口
```

[简明教程｜Linux中UFW的使用](https://zhuanlan.zhihu.com/p/98880088)
<!-- endtab -->
<!-- tab CentOS -->
```bash
firewall-cmd --zone=public --add-port=8050/tcp --permanent # 允许所有IP访问本机8050/tcp端口
firewall-cmd --reload # 需要重新加载才能生效
```
<!-- endtab -->
{% endtabs %}

### 关闭端口

{% tabs 关闭端口 %}
<!-- tab Ubuntu -->
删除规则

```bash
# sudo ufw delete <规则>
sudo ufw delete from 192.168.1.5 to any port 80 proto tcp # 删除之前所创建的192.168.1.5访问本机80/tcp端口规则
```

禁止端口

```bash
sudo ufw deny from 192.168.1.5 to any port 80 proto tcp # 禁止192.168.1.5访问本机80/tcp
```

一般配置，只需执行

```bash
sudo apt install ufw
sudo ufw enable
sudo ufw default deny
```

之后根据个人需求在开启相关访问权限就可以了。

<!-- endtab -->
<!-- tab CentOS -->
```bash
firewall-cmd --zone=public --remove-port=8050/tcp --permanent
firewall-cmd --reload # 需要重新加载才能生效
```

[IBM 学习: 使用 firewalld 构建 Linux 动态防火墙](https://www.ibm.com/developerworks/cn/linux/1507_caojh/index.html)
<!-- endtab -->
{% endtabs %}

## Linux设置静态IP

TODO

---

## Linux代理配置

使Linux能使用代理的方法有两种，一种为配置proxy环境变量，另一种为使用ProxyChains工具。

### 配置proxy环境变量

```bash
# 开启代理
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890 # 注意代理链接还是写的http
# 关闭代理
unset http_proxy
unset https_proxy
```

### ProxyChains

[ProxyChains](https://github.com/rofl0r/proxychains-ng)是一个开源的网络hook工具，支持scoks和HTTP代理

#### 安装

rivergold倾向于从源码编译安装

```bash
git clone https://github.com/rofl0r/proxychains-ng.git
```

```bash
cd proxychains-ng
./configure --prefix=/usr --sysconfdir=/etc
make
make install
make install-config
# cd .. && rm -rf proxychains-ng
```

#### 配置

编辑`/etc/proxychains.conf`

```bash
# <type> <ip> <port> [username] [password]
# socks
socks5 <ip> <port>
# http
http <ip> <port>
```

[Harker' Blog: Centos 7 安装 Proxychains 实现 Linux 代理](http://www.harker.cn/archives/proxychains.html)

---

## 基于Flask搭建简易文件服务器

由于经常需要对远程服务器进行上传和下载的操作，又不想安装过于复杂的软件，因此基于Flask及其插件卡开发了一个简易的文件服务器，基本满足日常需求。

基于Flask-AutoIndex可以使Flask支持文件目录展示功能，并可以进行下载

### 安装依赖

```bash
pip install flask flask_autoindex
```

文件结构如下:

```bash
flask_fileserver
├── __init__.py
└── route_func
    ├── __init__.py
    └── upload.py
```

完成的代码rivergold放在了[GitHub flask-fileserver](https://github.com/rivergold/flask-fileserver)上，需要的小伙伴可跳转参考，这里列出主要的两个文件，分别是`__init__.py`和`route_func/upload.py`。

```python
# __init__.py
# 用于创建app
import os.path
from flask import Flask
from flask_autoindex import AutoIndex
from .route_func.upload import upload_file


def create_app(browse_root_dir=None, upload_dir=None):
    app = Flask(__name__)
    AutoIndex(app, browse_root=browse_root_dir.as_posix())
    app.config['upload_dir'] = upload_dir
    app.add_url_rule('/upload',
                     endpoint='upload',
                     view_func=upload_file,
                     methods=['GET', 'POST'])
    return app
```

```python
# route_func/upload.py
# 提供文件上传和存储功能
from pathlib import Path
import flask
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', '.tar', '.zip'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(flask.current_app.config)
            save_dir = Path(flask.current_app.config['upload_dir'])
            save_path = save_dir / filename
            file.save(save_path.as_posix())
            return redirect(url_for('autoindex'))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
```

[stackoverflow: python flask browsing through directory with files](https://stackoverflow.com/a/58303738)
[Flask doc: Uploading Files](https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/)

## NFS远程挂载

基于 Linux 实现远程文件夹的挂载，可以把本地Linux的盘挂载到远程服务器上，从而使得二者间文件的读写更加方便。rivergold认为一个比较好的使用场景是，将本地的代码文件远程挂载到服务器上（或者是开发机上的代码文件挂载到另一个带有GPU的训练机器上），这样可以方便修改代码并进行实验。如果网络带宽够用，也可以把训练数据进行远程挂载。

NFS远程挂载配置，需要分别在Server端和Client端进行配置，Server端指的是提供盘的一方，Client端是执行挂载的一方。

### Server端

#### 安装

{% tabs 安装 %}
<!-- tab Ubuntu -->
```bash
sudo apt install nfs-kernel-server
```
<!-- endtab -->
<!-- tab CentOS -->
```bash
# 查看有无安装nfs-utils
rpm -qa | grep nfs-utils
rpm -qa | grep rpcbind
# 安装
yum -y install nfs-utils
yum -y install rpcbind
```
<!-- endtab -->
{% endtabs %}

#### 修改配置

编辑`/etc/exports`进行配置

```bash
# 格式说明
# <需要共享的文件夹路径> <Client的IP>(rw,no_root_squash,no_all_squash,async)
# Example
/data/share_dir 192.168.1.5(rw,no_root_squash,no_all_squash,async) # 共享目录/data/share_dir给192.168.1.5，192.168.1.5具有读写权限
```

#### 启动NFS

```bash
sudo systemctl start rpcbind
sudo systemctl start nfs
# sudo systemctl restart nfs-kernel-server
```

当`/etc/exports`发生修改后，可以使用以下命令在不重启NFS服务的情况下使新配置的生效

```bash
exportfs -a
```

[博客园: Linux 下配置 nfs 并远程挂载](https://www.cnblogs.com/freeweb/p/6593861.html)

如果是CentOS的话，需要开启相关端口

```bash
firewall-cmd --permanent --zone=public --add-port=111/tcp
firewall-cmd --permanent --zone=public --add-port=111/udp
firewall-cmd --permanent --zone=public --add-port=2049/tcp
firewall-cmd --permanent --zone=public --add-port=2049/udp
firewall-cmd --reload
```

[StackExchange-serverfault: Which ports do I need to open in the firewall to use NFS?](https://serverfault.com/questions/377170/which-ports-do-i-need-to-open-in-the-firewall-to-use-nfs)

### Client

#### 安装

{% tabs 安装 %}
<!-- tab Ubuntu -->
```bash
sudo apt install nfs-common
```
<!-- endtab -->
<!-- tab CentOS -->
```bash
yum -y install nfs-utils
```
<!-- endtab -->
{% endtabs %}

#### 挂载

```bash
sudo mount -t nfs <server_ip>:<server_folder_path> <local_mount_path>
```

#### 卸载

```bash
sudo umount <local_mount_path>
# 强制卸载
sudo umount -f -l <local_mount_path>
```

- `-f`: Force unmount (in case of an unreachable NFS system). (Requires kernel 2.1.116 or later.)
- `-l`: Lazy unmount. Detach the filesystem from the filesystem hierarchy now, and cleanup all references to the filesystem as soon as it is not busy anymore. (Requires kernel 2.4.11 or later.)

[stackoverflow: How to unmount NFS when server is gone?](https://askubuntu.com/questions/292043/how-to-unmount-nfs-when-server-is-gone)

## :fallen_leaf:Flask

- [Flask 中文](https://dormousehole.readthedocs.io/en/latest/)

### Example

```python
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def get_callback_data():
    print('##Debug## method is {}'.format(request.method))
    print(request.get_json())
      if request.method == 'GET':
          print(request.form['data'])
          return 'ok'
      elif request.method == 'POST':
          print(request.form['data'])
          return 'ok'
      else:
          raise ValueError('Only support GET or POST method.')
```

Run:

```bash
export FLASK_APP=<python_script_name>
flask run --host=0.0.0.0
```

**_References:_**

- [Flask Doc: 快速上手](https://dormousehole.readthedocs.io/en/latest/quickstart.html)

### Get request data

```python
data = request.get_json()
```

**_References:_**

- [stackoverflow: How to get data received in Flask request](https://stackoverflow.com/questions/10434599/how-to-get-data-received-in-flask-request)

---

### Params with url

```python
@app.route('/param', methods=['GET', 'POST'])
def get_callback_data(param):
    print(param)
```

**_References:_**

- [CSDN: Flask 带参 URL 传值的方法](https://blog.csdn.net/weixin_36380516/article/details/80008496)

## :fallen_leaf:科学上网

关于科学上网，折腾了很久，搭建过 shaodowsocks 和 V2ray，使用过 Vultr 和 Linode。整体速度也都还行，但是由于需要花费一定时间维护（大部分时间是 IP 被封了换机器）。最终我决定还是投入到[just my socks](https://justmysocks.net/)的怀抱，不折腾了。

### Tools

#### SwitchyOmega

一款好用的用于管理 proxy 代理规则的浏览器插件

相应的代理规则可以从[GFWList](https://github.com/gfwlist/gfwlist)找到更新链接。

References:

- [MDN web docs: Client-Server overview
  ](https://developer.mozilla.org/zh-CN/docs/learn/Server-side/First_steps/Client-Server_overview)
- [Microservices](https://www.redhat.com/zh/topics/microservices/what-are-microservices)
- [redhat: 什么是微服务？](https://www.redhat.com/zh/topics/microservices/what-are-microservices)
- [Ubuntu 20.04 中配置NFS服务](https://www.linuxprobe.com/ubuntu-configure-nfs.html)
