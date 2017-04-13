## Install Spark
1. Install Java(jre and jdk)
    ```shell
    sudo apt-add-repository ppa:webupd8team/java  
    sudo apt-get update  
    sudo apt-get install oracle-java8-installer  
    ```

2. Install SBT<br>
    SBT is an open source build tool for Scala and Java projects.
    ```shell
    echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list  
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823  
    sudo apt-get update  
    sudo apt-get install sbt  
    ```

3. Download Spark<br>
    Download Spark from [official website](http://spark.apache.org/downloads.html). Generally, we download the pre-build version.Then decompress it into the path you want using
    ```shell
    tar xzvf spark-2.0.1-bin-hadoop2.7.tgz  
    mv spark-2.0.1-bin-hadoop2.7/ spark  
    sudo mv spark/ <spark_path, here we use /usr/lib/>  
    ```

4. Configure Spark<br>
    ```shell
    cd /usr/lib/conf/  
    cp spark-env.sh.template spark-env.sh  
    nano spark-env.sh  
    ```
    Add the following lines<br>
    ```
    JAVA_HOME=/usr/lib/jvm/java-8-oracle  
    SPARK_WORKER_MEMORY=4g  
    ```

5. Configure .bashrc<br>
    ```shell
    sudo nano ~/.bashrc
    export JAVA_HOME=/usr/lib/jvm/java-8-oracle  
    export SBT_HOME=/usr/share/sbt-launcher-packaging/bin/sbt-launch.jar  
    export SPARK_HOME=/usr/lib/
    export PATH=$PATH:$JAVA_HOME/bin  
    export PATH=$PATH:$SBT_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin  
    ```

    Configure PYTHONPATH, in order to import `pyspark` in python script or shell

    ```shell
    export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
    ```

    Update .bashrc
    ```shell
    . ~/.bashrc
    ```
- Reference
    - [Installing Apache Spark on Ubuntu 16.04](https://www.santoshsrinivas.com/installing-apache-spark-on-ubuntu-16-04/)
    - [Stackoverflow: importing pyspark in python shell](http://stackoverflow.com/questions/23256536/importing-pyspark-in-python-shell)

## Basic Understanding
- Spark is compatible with Hadoop

## Problems and Solutions
- What is the difference between JDK and JRE? ([ref][ref_1])
    - JRE: Java Runtime Environment. It is basically the Java Virtual Machine where your Java programs run on. It also includes browser plugins for Applet execution.

    - JDK: It's the full featured Software Development Kit for Java, including JRE, and the compilers and tools (like JavaDoc, and Java Debugger) to create and compile programs.

- Vim do not work well in Conemu for docker container?
    - When I start vim in a docker container in ConEmu, vim do not work well when I move cursor.
    - I solute it by update Docker for Windows from 1.13 to 1.17. The author of ConEmu said that this bug is not from ConEmu, it is because the old version docker not support it well. And the latest version of docker fix this bug.

## 我的Spark学习笔记
Spark
: 一个用来实现快速、通用的集群计算的平台

RDD(Resilient Distributed Dataset)
: 弹性分布式数据集， 表示分布在多个计算节点上可以并行操作的元素集合，是Spark主要的编程抽象。
