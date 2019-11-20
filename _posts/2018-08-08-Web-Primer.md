# :fallen_leaf:Basic

## Server & Client

- [MDN web docs: Client-Server overview
  ](https://developer.mozilla.org/zh-CN/docs/learn/Server-side/First_steps/Client-Server_overview)

---

## TCP

TCP client not need to specify the port.

**理解：** TCP client 不需要指定端口号

## Microservices(微服务)

What is [**Microservices**](https://www.redhat.com/zh/topics/microservices/what-are-microservices)?

**_References:_**

- [redhat: 什么是微服务？](https://www.redhat.com/zh/topics/microservices/what-are-microservices)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Common Commands

## Config firewall to expose port

### CentOS: `firewall-cmd`

When change config of `firewall-cmd`, you need to run `firewall-cmd --reload` to make it take effect.

- Install

  ```bash
  yum install firewalld firewall-config
  ```

- Start service

  ```bash
  systemctl enable firewalld.service
  systemctl start firewalld.service
  ```

- Check firewall statUS

  ```bash
  systemctl status firewalld
  ```

- List all rules

  ```bash
  firewall-cmd --list-all
  ```

- Open port

  ```bash
  firewall-cmd --zone=public --add-port=8050/tcp --permanent
  ```

- Remove port

  ```bash
  firewall-cmd --zone=public --remove-port=8050/tcp --permanent
  ```

Ref [IBM 学习: 使用 firewalld 构建 Linux 动态防火墙](https://www.ibm.com/developerworks/cn/linux/1507_caojh/index.html)

**_References:_**

- [StackExchange serverfault: How to remove access to a port using firewall on Centos7?](https://serverfault.com/a/865041)

### Ubuntu: `ufw`

When added rule into ufw, ufw doest not need to reload to take effect.

- Install

  ```bash
  sudo apt install ufw
  ```

- Enable ufw

  ```bash
  sudo ufw enable
  ```

- Disable ufw

  ```bash
  sudo ufw disable
  ```

- Check the default configuration

  ```bash
  sudo ufw show raw
  ```

- Allow connections to SSH

  ```bash
  sudo ufw allow ssh
  # or
  sudo ufw allow 22/tcp
  ```

- Enable other services

  ```bash
  sudo ufw allow 80/tcp
  sudo ufw allow 443/tcp
  sudo ufw allow 21/tcp
  ```

- Allow connections from specific IP addresses

  ```bash
  sudo ufw allow from 111.111.111.111
  ```

  Check the status:

  ```bash
  sudo ufw status
  ```

  Delete the rule:

  ```bash
  sudo ufw delete allow from 111.111.111.111
  ```

- Allow connections from specific IP addresses to specific port

  ```bash
  sudo ufw allow <source ip> to any port <destination port>
  ```

Ref [RoseHosting: How To Set Up a Firewall with UFW on Ubuntu 16.04](https://www.rosehosting.com/blog/set-up-firewall-with-ufw-on-ubuntu/)

**_References:_**

- [DigitalOcean: UFW Essentials: Common Firewall Rules and Commands](https://www.digitalocean.com/community/tutorials/ufw-essentials-common-firewall-rules-and-commands)

<!--  -->
<br>

---

<!--  -->

## curl

### curl post

```bash
curl -d "data=test" <http_address>
```

**_Ref:_** [简书: CURL 命令模拟 Http Get/Post 请求](https://www.jianshu.com/p/a8b648e96753)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Setup Apache File Server

## Setup a HTTP Server

- [APACHE HTTP Server Documentation Compiling and Installing](http://httpd.apache.org/docs/current/install.html#customize)

---

## Install

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

## Config

**Ubuntu**

Edit `/etc/apache2/sites-available/000-default.conf`

Change `DocumentRoot` to your path.

```conf
DocumentRoot /path/to/my/project
```

Then, run `sudo systemctl restart apache2` to restart apache2 service.

**_References:_**

- :thumbsup:[stackoverflow: Change Apache document root folder to secondary hard drive](https://askubuntu.com/a/738527)

---

## Problem & Solution

### When visit apache web, occur `Forbidden You don't have permission to access / on this server`

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

---

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

Change http work folder `DocumentRoot`

1. vim `/etc/httpd/conf/httpd.conf`

   ```vim
   #
   121 # DocumentRoot "/var/www/html"
   122 DocumentRoot "/data/www"
   #
   131 # <Directory "/var/www">
   132 <Directory "/data/www">
   #
   141 #<Directory "/var/www/html">
   142 <Directory "/data/www">
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

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:NFS Mount Remote folder

- Server: `CentOS 7`
- Client: `Ubuntu 18.04`

## Server

1. **Check & Install dependences**

   ```bash
   rpm -qa | grep nfs-utils
   rpm -qa | grep rpcbind
   ```

   if not occur the packages, please install them:

   ```bash
   yum -y install nfs-utils
   yum -y install rpcbind
   ```

2. **Edit `/etc/exports` to config access permission**

   ```bash
   <folder path you want to share> <ip of client>(rw,no_root_squash,no_all_squash,async)
   ```

   e.g.

   ```bash
   /nfs_test 192.168.1.5(rw,no_root_squash,no_all_squash,async)
   ```

3. **Config firewall**

   NFS server need port 111 (TCP and UDP), port 2049 (TCP and UDP)

   ```bash
   firewall-cmd --permanent --zone=public --add-port=111/tcp
   firewall-cmd --permanent --zone=public --add-port=111/udp
   firewall-cmd --permanent --zone=public --add-port=2049/tcp
   firewall-cmd --permanent --zone=public --add-port=2049/udp
   firewall-cmd --reload
   ```

   Ref [StackExchange-serverfault: Which ports do I need to open in the firewall to use NFS?](https://serverfault.com/questions/377170/which-ports-do-i-need-to-open-in-the-firewall-to-use-nfs)

4. **Start service**

   ```bash
   systemctl start rpcbind
   systemctl start nfs
   ```

5. **If changed the config, refresh**

   ```bash
   exportfs -a
   ```

Ref [博客园: Linux 下配置 nfs 并远程挂载](https://www.cnblogs.com/freeweb/p/6593861.html)

---

## Client

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

Ref [CSDN: Ubuntu NFS 服务器客户端配置方法](https://blog.csdn.net/zhuxiaoping54532/article/details/53435158)

**_References:_**

- [Howtoing 运维教程: 如何在 Ubuntu 18.04 上设置 NFS 挂载](https://www.howtoing.com/how-to-set-up-an-nfs-mount-on-ubuntu-18-04)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Flask

- [Flask 中文](https://dormousehole.readthedocs.io/en/latest/)

## Example

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

**_Ref:_** [Flask Doc: 快速上手](https://dormousehole.readthedocs.io/en/latest/quickstart.html)

---

## Get request data

```python
data = request.get_json()
```

**_Ref:_** [stackoverflow: How to get data received in Flask request](https://stackoverflow.com/questions/10434599/how-to-get-data-received-in-flask-request)

---

## Params with url

```python
@app.route('/param', methods=['GET', 'POST'])
def get_callback_data(param):
    print(param)
```

**_References:_**

- [CSDN: Flask 带参 URL 传值的方法](https://blog.csdn.net/weixin_36380516/article/details/80008496)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:翻越长城墙

## v2ray

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

---

## GFWList

- [GFWList](https://github.com/gfwlist/gfwlist)
