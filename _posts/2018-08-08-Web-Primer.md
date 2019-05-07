# Basic understanding

- [MDN web docs: Client-Server overview
](https://developer.mozilla.org/zh-CN/docs/learn/Server-side/First_steps/Client-Server_overview)


# Apache

## Build a HTTP Server

- [APACHE HTTP Server Documentation Compiling and Installing](http://httpd.apache.org/docs/current/install.html#customize)

### Ubuntu

#### Install

***References:***

- [askubuntu: How to set up a simple file server?](https://askubuntu.com/questions/556858/how-to-set-up-a-simple-file-server)
- [DigitalOcean: How To Install the Apache Web Server on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-16-04)
- [How To Configure the Apache Web Server on an Ubuntu or Debian VPS](https://www.digitalocean.com/community/tutorials/how-to-configure-the-apache-web-server-on-an-ubuntu-or-debian-vps)
- [知乎: 实现一个http服务器需要怎样进行？需要哪些知识呢？](https://www.zhihu.com/question/20199473)
- [简书: Linux搭建简单的http文件服务器](https://www.jianshu.com/p/e1a6219167cf)
- [Blog: Ubuntu下Apache服务器的配置](https://www.ezlippi.com/blog/2016/01/apache-configuration-in-ubuntu.html)

#### Apache Config

After you change the apache config, you'd have to run `sudo service apache2 restart` to restart apache to allow config update.

#### Change listen port

Change listen port in `/etc/apache2/ports.conf` and `/etc/apache2/sites-enabled/000-default.conf`

```bash
sudo vim /etc/apache2/ports.conf
sudo vim /etc/apache2/sites-enabled/000-default.conf
```

***References:***

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

***References:***

- [TecMint: How to Change Default Apache ‘DocumentRoot’ Directory in Linux](https://www.tecmint.com/change-root-directory-of-apache-web-server/)
- [CSDN: Apache禁用测试页（默认页）](https://blog.csdn.net/Aguangg_6655_la/article/details/53915917)

### Problems & Solution

#### Show index not default index html

***References:***

- [askubuntu: Apache shows index of/ instead of default index html [duplicate]](https://askubuntu.com/questions/450211/apache-shows-index-of-instead-of-default-index-html)

#### If ohter computer in LAN cannot access to the http, but your computer can access. Maybe you need to set `firewall` to allow the port for `Apache`

```bash
sudo ufw allow <your listen port>
```

***References:***

- [Blog: Ubuntu默认防火墙安装、启用、配置、端口、查看状态相关信息](https://www.cnblogs.com/toughlife/p/5475615.html)

<!--  -->
<br>

***

<br>
<!--  -->

# Basics

## TCP

TCP client not need to specify the port.

**理解：** TCP client不需要指定端口号

## Microservices(微服务)

What is [**Microservices**](https://www.redhat.com/zh/topics/microservices/what-are-microservices)?

***References:***

- [redhat: 什么是微服务？](https://www.redhat.com/zh/topics/microservices/what-are-microservices)

<!--  -->
<br>

***

<br>
<!--  -->

# 翻越长城墙

## v2ray

### 搭建

Ref [Blog: 轻松搭建和配置V2Ray](https://mianao.info/2018/04/23/%E8%BD%BB%E6%9D%BE%E6%90%AD%E5%BB%BA%E5%92%8C%E9%85%8D%E7%BD%AEv2ray)

#### 多人使用

Ref [Github v2ray/v2ray-core: V2Ray多用户配置的正确姿势究竟是怎样？ #679](https://github.com/v2ray/v2ray-core/issues/679)

### 客户端

***References:***

- [Blog: v2ray的第三方客户端](http://briteming.hatenablog.com/entry/2017/10/21/124645)

#### Linux

Ref [Linux 系统下v2ray客户端使用](https://octopuspalm.top/2018/08/18/Linux%20%E7%B3%BB%E7%BB%9F%E4%B8%8Bv2ray%E5%AE%A2%E6%88%B7%E7%AB%AF%E4%BD%BF%E7%94%A8/)

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