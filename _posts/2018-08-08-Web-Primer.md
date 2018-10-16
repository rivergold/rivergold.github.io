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

<br>

***

<br>

# TCP

## Basics

TCP client not need to specify the port.

**理解：** TCP client不需要指定端口号