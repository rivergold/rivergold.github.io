# CGDB

- [cgdb](http://cgdb.github.io/)

- [知乎： 终端调试哪家强？](https://zhuanlan.zhihu.com/p/32843449)

## Install

### Ubuntu

1. Get latest sources

    ```bash
    git clone git://github.com/cgdb/cgdb.git
    cd cgdb
    ./autogen.sh
    ```

2. Install dependences

    ```bash
    sudo apt install libncurses5-dev flex texinfo libreadline-dev
    ```

3. Genrate makefile

    ```bash
    ./configure --prefix=<path you want to install>
    ```

4. Build and install

    ```bash
    make -j8
    make install
    ```

***References:***

- [CSDN: linux安装cgdb](https://blog.csdn.net/analogous_love/article/details/53389070)

## Use

***Ref:*** [Github: cgdb-manual-in-chinese](https://github.com/leeyiw/cgdb-manual-in-chinese)