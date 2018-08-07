# Apache

## Build a Simple Web File Server

### Install

***References:***

- [askubuntu: How to set up a simple file server?](https://askubuntu.com/questions/556858/how-to-set-up-a-simple-file-server)
- [DigitalOcean: How To Install the Apache Web Server on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-16-04)
- [How To Configure the Apache Web Server on an Ubuntu or Debian VPS](https://www.digitalocean.com/community/tutorials/how-to-configure-the-apache-web-server-on-an-ubuntu-or-debian-vps)
- [知乎: 实现一个http服务器需要怎样进行？需要哪些知识呢？](https://www.zhihu.com/question/20199473)
- [简书: Linux搭建简单的http文件服务器](https://www.jianshu.com/p/e1a6219167cf)
- [Blog: Ubuntu下Apache服务器的配置](https://www.ezlippi.com/blog/2016/01/apache-configuration-in-ubuntu.html)

### Apache Config

#### Change listen port

Change listen port in `/etc/apache2/ports.conf` and `/etc/apache2/sites-enabled/000-default.conf`

```bash
sudo vim /etc/apache2/ports.conf
sudo vim /etc/apache2/sites-enabled/000-default.conf
```

***References:***

- [TecMint: How to Change Apache HTTP Port in Linux](https://www.tecmint.com/change-apache-port-in-linux/)
- [OSTechNix: How To Change Apache Default Port To A Custom Port](https://www.ostechnix.com/how-to-change-apache-ftp-and-ssh-default-port-to-a-custom-port-part-1/)

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