---
title: "Basics Web Usage"
last_modified_at: 2020-02-22
categories:
  - Memo
tags:
  - Linux
  - Tool
---

Some tools and tricks about web.

## :fallen_leaf:Basic

### Server & Client

- [MDN web docs: Client-Server overview
  ](https://developer.mozilla.org/zh-CN/docs/learn/Server-side/First_steps/Client-Server_overview)

### TCP

TCP client not need to specify the port.

> @rivergold: 在创建 TCP 连接时，客户端不需要 TCP 不需要指定端口号

### Microservices(微服务)

What is [**Microservices**](https://www.redhat.com/zh/topics/microservices/what-are-microservices)?

**_References:_**

- [redhat: 什么是微服务？](https://www.redhat.com/zh/topics/microservices/what-are-microservices)

## :fallen_leaf:Firewall Config Command

### CentOS `firewall-cmd`

#### Install

```shell
yum install firewalld firewall-config
```

#### Start service

```shell
systemctl enable firewalld.service
systemctl start firewalld.service
```

#### Check status

```shell
systemctl status firewalld
```

#### List all rules

```shell
firewall-cmd --list-all
```

#### Open port

```shell
firewall-cmd --zone=public --add-port=8050/tcp --permanent
```

#### Close port

```shell
firewall-cmd --zone=public --remove-port=8050/tcp --permanent
```

**:triangular_flag_on_post:When change config of `firewall-cmd`, you need to run `firewall-cmd --reload` to make it take effect.**

**_References:_**

[IBM 学习: 使用 firewalld 构建 Linux 动态防火墙](https://www.ibm.com/developerworks/cn/linux/1507_caojh/index.html)

- [StackExchange serverfault: How to remove access to a port using firewall on Centos7?](https://serverfault.com/a/865041)

### Ubuntu `ufw`

#### Install

```shell
sudo apt install ufw
```

#### Enable ufw

```shell
sudo ufw enable
```

#### Disable ufw

```shell
sudo ufw disable
```

#### Check the default configuration

```shell
sudo ufw show raw
```

#### All connections to SSH

```shell
sudo ufw allow ssh
# or
sudo ufw allow 22/tcp
```

#### Enable other services

```shell
# Some examples
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 21/tcp
```

#### Allow connection from specific IP address

```shell
sudo ufw allow from <from_ip>
# Allow connection from specific IP addresses to specific port
sudo ufw allow <from_ip> to any port <to_port>
```

Check status

```shell
sudo ufw status
```

Delete rule

```shell
sudo ufw delete from <from_ip>
```

**_References:_**

- [RoseHosting: How To Set Up a Firewall with UFW on Ubuntu 16.04](https://www.rosehosting.com/blog/set-up-firewall-with-ufw-on-ubuntu/)
- [DigitalOcean: UFW Essentials: Common Firewall Rules and Commands](https://www.digitalocean.com/community/tutorials/ufw-essentials-common-firewall-rules-and-commands)

**When added rule into ufw, ufw doest not need to reload to take effect.**

## :fallen_leaf:Nginx File Server

使用 Nginx 搭建文件服务器，方便查看远程服务器上的实验结果。

**注:** Nginx 也是支持上传的，只是我现在搭建的还不够好。

### Install

**CentOS**

```shell
yum install nginx
```

**Ubuntu**

```shell
sudo apt install nginx
```

**Build from source**

Download Source from [nginx.org](http://nginx.org/en/download.html)

Build

```shell
./configure \
--prefix=/root/software/tool/nginx/nginx-1.16.1/install \
--with-http_ssl_module \
--add-module=/root/software/tool/nginx/nginx-upload-module-2.2
```

**_References:_**

- [segmentfault: 编译安装 Nginx 1.14](https://segmentfault.com/a/1190000015992091)

#### During build occur `fatal error: md5.h: No such file or directory`

Downloaded `ngx_http_upload_module` source code has bug, need to change other version.

```shell
wget https://github.com/Austinb/nginx-upload-module/archive/2.2.zip
```

**_References:_**

- [SundayLE Blog: Nginx 编译](https://www.sundayle.com/nginx/)

#### make install occur `make: 'install' is up to date.`

Some file or dictionary has same name, you need to delete them.

**_References:_**

- [CSDN: makefile 出现“is up to date”提示的修改方法](https://blog.csdn.net/beizhetaiyangxingzou/article/details/39967149)

### Start Service

```shell
nginx
```

### Stop Service

```shell
killall nginx
```

### Config

Config file path is `/etc/nginx/nginx.conf`

1. Check nginx user

   ```shell
   ps aux | grep "nginx: worker process"
   ```

   **_References:_**

   - [CSDN: 解决 Nginx 出现 403 forbidden (13: Permission denied)报错的四种方法](https://blog.csdn.net/onlysunnyboy/article/details/75270533)

2. Set user

   ```shell
   user root root;
   ```

3. Inspect permission

   ```shell
   ll your_path
   chown -R root:root <your_path>
   chmod -R 755 <your_path>
   ```

4. Edit `nginx.conf`

   ```shell
   http {
        log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                        '$status $body_bytes_sent "$http_referer" '
                        '"$http_user_agent" "$http_x_forwarded_for"';

        access_log  /var/log/nginx/access.log  main;

        sendfile            on;
        tcp_nopush          on;
        tcp_nodelay         on;
        keepalive_timeout   65;
        types_hash_max_size 2048;

        include             /etc/nginx/mime.types;
        default_type        application/octet-stream;

        # Load modular configuration files from the /etc/nginx/conf.d directory.
        # See http://nginx.org/en/docs/ngx_core_module.html#include
        # for more information.
        include /etc/nginx/conf.d/*.conf;

        # ----------You need to edit-----------
        charset utf-8,gbk; # 开始中文支持
        autoindex_exact_size off; # 文件大小显示为Mb代替bytes
        # ----------You need to edit-----------

   server {
        listen       80 default_server;
        listen       [::]:80 default_server;
        server_name  _;
        root         /usr/share/nginx/html;

        # Load configuration files for the default server block.
        include /etc/nginx/default.d/*.conf;

        # ----------You need to edit-----------
        location / {
              root /data;
              autoindex on; # 显示root下的文件目录
        }
        # ----------You need to edit-----------
   ```

**_References:_**

- :thumbsup:[博客园: 关于 Nginx 403 forbidden 错误踩的坑 directory index of "/xx/xx/xx/" is forbidden](https://www.cnblogs.com/Cong0ks/p/11958846.html)
- [Blog: nginx 设置目录浏览及中文乱码问题解决](https://wangheng.org/nginx-set-directory-browsing-and-solve-the-problem-of-chinese-garbled.html)

## :fallen_leaf:Setup Apache File Server

Apache 是我在使用 Nginx 之前所使用的，Apache 相比于 Nginx 要更庞大一点，而我的需求仅是简单的文件传输，所以便更换成了 Nginx。

### Setup a HTTP Server

- [APACHE HTTP Server Documentation Compiling and Installing](http://httpd.apache.org/docs/current/install.html#customize)

### Install

**Ubuntu**

```shell
sudo apt install apache2
```

**CentOS**

```shell
sudo yum install httpd
sudo systemctl enable httpd
sudo systemctl start httpd
```

### Config

**Ubuntu**

Edit `/etc/apache2/sites-available/000-default.conf`

Change `DocumentRoot` to your path.

```conf
DocumentRoot /path/to/my/project
```

Then, run `sudo systemctl restart apache2` to restart apache2 service.

**_References:_**

- :thumbsup:[stackoverflow: Change Apache document root folder to secondary hard drive](https://askubuntu.com/a/738527)

### Problem & Solution

#### When visit apache web, occur `Forbidden You don't have permission to access / on this server`

This is because the `Require` is `denied`.

**Solution**

Edit `/etc/apache2`

```conf
<Directory />
    Options Indexes FollowSymLinks Includes ExecCGI
    AllowOverride All
    Require all granted
</Directory>
```

### Ubuntu

#### Install

**_References:_**

- [askubuntu: How to set up a simple file server?](https://askubuntu.com/questions/556858/how-to-set-up-a-simple-file-server)
- [DigitalOcean: How To Install the Apache Web Server on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-16-04)
- [How To Configure the Apache Web Server on an Ubuntu or Debian VPS](https://www.digitalocean.com/community/tutorials/how-to-configure-the-apache-web-server-on-an-ubuntu-or-debian-vps)
- [知乎: 实现一个 http 服务器需要怎样进行？需要哪些知识呢？](https://www.zhihu.com/question/20199473)
- [简书: Linux 搭建简单的 http 文件服务器](https://www.jianshu.com/p/e1a6219167cf)
- [Blog: Ubuntu 下 Apache 服务器的配置](https://www.ezlippi.com/blog/2016/01/apache-configuration-in-ubuntu.html)

#### Apache Config

After you change the apache config, you'd have to run `sudo service apache2 restart` to restart apache to allow config update.

#### Change listen port

Change listen port in `/etc/apache2/ports.conf` and `/etc/apache2/sites-enabled/000-default.conf`

```bash
sudo vim /etc/apache2/ports.conf
sudo vim /etc/apache2/sites-enabled/000-default.conf
```

**_References:_**

- [TecMint: How to Change Apache HTTP Port in Linux](https://www.tecmint.com/change-apache-port-in-linux/)
- [OSTechNix: How To Change Apache Default Port To A Custom Port](https://www.ostechnix.com/how-to-change-apache-ftp-and-ssh-default-port-to-a-custom-port-part-1/)

### CentOS

#### Install

```shell
sudo yum install httpd
sudo systemctl enable httpd
sudo systemctl start httpd
```

#### Configure

TODO: update

1. Edit `/etc/httpd/conf/httpd.conf`

   ```shell
   # Before
   <Directory "/var/www">
       AllowOverride None
       # Allow open access:
       Require all granted
   </Directory>

   # Further relax access to the default document root:
   <Directory "/var/www/html">

   # After
   <Directory "your_path">
       AllowOverride None
       # Allow open access:
       Require all granted
   </Directory>
   # Further relax access to the default document root:
   <Directory "your_path">
   ```

2. vim `/etc/httpd/conf.d/welcome.conf`
   Comment all lines in this file.

**_References:_**

- [TecMint: How to Change Default Apache ‘DocumentRoot’ Directory in Linux](https://www.tecmint.com/change-root-directory-of-apache-web-server/)
- [CSDN: Apache 禁用测试页（默认页）](https://blog.csdn.net/Aguangg_6655_la/article/details/53915917)

### Problems & Solution

#### Show index not default index html

**_References:_**

- [askubuntu: Apache shows index of/ instead of default index html [duplicate]](https://askubuntu.com/questions/450211/apache-shows-index-of-instead-of-default-index-html)

#### If ohter computer in LAN cannot access to the http, but your computer can access. Maybe you need to set `firewall` to allow the port for `Apache`

```bash
sudo ufw allow <your listen port>
```

**_References:_**

- [Blog: Ubuntu 默认防火墙安装、启用、配置、端口、查看状态相关信息](https://www.cnblogs.com/toughlife/p/5475615.html)

## :fallen_leaf:NFS Mount Remote folder

基于 Linux 实现远程文件夹的挂载，从而实验远程文件的编辑。但使用体验的话还是偏慢，感觉不是很方便。这里就记录一下，但不推荐使用。

- Server: `CentOS 7`
- Client: `Ubuntu 18.04`

### Server

1. **Check & Install dependences**

   ```shell
   rpm -qa | grep nfs-utils
   rpm -qa | grep rpcbind
   ```

   if not occur the packages, please install them:

   ```shell
   yum -y install nfs-utils
   yum -y install rpcbind
   ```

2. **Edit `/etc/exports` to config access permission**

   ```shell
   <folder path you want to share> <ip of client>(rw,no_root_squash,no_all_squash,async)
   ```

   e.g.

   ```shell
   /nfs_test 192.168.1.5(rw,no_root_squash,no_all_squash,async)
   ```

3. **Config firewall**

   NFS server need port 111 (TCP and UDP), port 2049 (TCP and UDP)

   ```shell
   firewall-cmd --permanent --zone=public --add-port=111/tcp
   firewall-cmd --permanent --zone=public --add-port=111/udp
   firewall-cmd --permanent --zone=public --add-port=2049/tcp
   firewall-cmd --permanent --zone=public --add-port=2049/udp
   firewall-cmd --reload
   ```

   Ref [StackExchange-serverfault: Which ports do I need to open in the firewall to use NFS?](https://serverfault.com/questions/377170/which-ports-do-i-need-to-open-in-the-firewall-to-use-nfs)

4. **Start service**

   ```shell
   systemctl start rpcbind
   systemctl start nfs
   ```

5. **If changed the config, refresh**

   ```shell
   exportfs -a
   ```

Ref [博客园: Linux 下配置 nfs 并远程挂载](https://www.cnblogs.com/freeweb/p/6593861.html)

### Client

1. **Install**

   ```bash
   sudo apt install nfs-common
   ```

2. **Mount**

   ```bash
   sudo mount -t nfs <server ip>:<server folder path> <local mount path>
   ```

3. **Umount**

   ```bash
   sudo umount <local mount path>
   ```

**_References:_**

- [CSDN: Ubuntu NFS 服务器客户端配置方法](https://blog.csdn.net/zhuxiaoping54532/article/details/53435158)
- [Howtoing 运维教程: 如何在 Ubuntu 18.04 上设置 NFS 挂载](https://www.howtoing.com/how-to-set-up-an-nfs-mount-on-ubuntu-18-04)

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

<!-- ### v2ray

### Install

**Server**

```shell
bash <(curl -s -L https://git.io/v2ray.sh)
```

- `TCP`
- My default port `58888`

Open firewall for your v2ray port.

**_Ref:_** [233boy: V2Ray 搭建详细图文教程](https://github.com/233boy/v2ray/wiki/V2Ray%E6%90%AD%E5%BB%BA%E8%AF%A6%E7%BB%86%E5%9B%BE%E6%96%87%E6%95%99%E7%A8%8B)

**Client**

Download `v2ray-core` from [Github](https://github.com/v2ray/v2ray-core/releases).

`Config.json` example:

```json
{
  "log": {
    "loglevel": "info"
  },
  "inbounds": [
    {
      // 本地代理
      "port": 1089,
      "protocol": "socks",
      "sniffing": {
        "enabled": true,
        "destOverride": ["http", "tls"]
      },
      "settings": {
        "udp": true // 开启 UDP 协议支持
      }
    },
    {
      "port": 8080,
      "protocol": "http",
      "sniffing": {
        "enabled": true,
        "destOverride": ["http", "tls"]
      }
    }
  ],
  "outbounds": [
    {
      "tag": "proxy-vmess",
      "protocol": "vmess",
      "settings": {
        "vnext": [
          {
            "address": "<>", // 服务器的 IP
            "port": <>, // 服务器的端口
            "users": [
              {
                // id 就是 UUID，相当于用户密码
                "id": "<>",
                "alterId": 4
              }
            ]
          }
        ]
      }
    },
    {
      "tag": "direct",
      "settings": {},
      "protocol": "freedom"
    }
  ],
  "dns": {
    "server": ["8.8.8.8", "1.1.1.1"],
    // 你的 IP 地址，用于 DNS 解析离你最快的 CDN
    "clientIp": "203.208.40.63"
  },
  // 配置路由功能，绕过局域网和中国大陆地址
  "routing": {
    "domainStrategy": "IPOnDemand",
    "rules": [
      {
        "type": "field",
        "domain": [
          // 默认跳过国内网站，如果想要代理某个国内网站可以添加到下列列表中
          "cnblogs.com"
        ],
        "outboundTag": "proxy-vmess"
      },
      {
        "type": "field",
        "domain": ["geosite:cn"],
        "outboundTag": "direct"
      },
      {
        "type": "field",
        "outboundTag": "direct",
        "ip": ["geoip:cn", "geoip:private"]
      }
    ]
  }
}
```

**_Ref:_** [YEARLINY: 面向新手的 V2Ray 搭建指南](https://yuan.ga/v2ray-build-guide-for-novices/)

### 搭建

Ref [Blog: 轻松搭建和配置 V2Ray](https://mianao.info/2018/04/23/%E8%BD%BB%E6%9D%BE%E6%90%AD%E5%BB%BA%E5%92%8C%E9%85%8D%E7%BD%AEv2ray)

#### 多人使用

Ref [Github v2ray/v2ray-core: V2Ray 多用户配置的正确姿势究竟是怎样？ #679](https://github.com/v2ray/v2ray-core/issues/679)

### 客户端

**_References:_**

- [Blog: v2ray 的第三方客户端](http://briteming.hatenablog.com/entry/2017/10/21/124645)

#### Linux

Ref [Linux 系统下 v2ray 客户端使用](https://octopuspalm.top/2018/08/18/Linux%20%E7%B3%BB%E7%BB%9F%E4%B8%8Bv2ray%E5%AE%A2%E6%88%B7%E7%AB%AF%E4%BD%BF%E7%94%A8/)

```bash
# 后台启动
nohup ./v2ray &
# 关闭
pkill v2ray
```

Ref [makeuseof: 7 Different Ways To Kill Unresponsive Programs in Linux](https://www.makeuseof.com/tag/6-different-ways-to-end-unresponsive-programs-in-linux/)

Error `nohup: failed to run command.:Permission denied`

You can't call `nohup` on a shell construct such as alias, function or buildin. `nohup ./test.sh` is the correnct way to run.

Ref [StackExchange: nohup: failed to run command `.': Permission denied](https://unix.stackexchange.com/questions/386545/nohup-failed-to-run-command-permission-denied)
 -->
