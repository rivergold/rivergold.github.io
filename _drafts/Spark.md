Spark

# Preprepares
- Java
- Anaconda
- Vim

# Install Spark
1. Install Java(jre and jdk)
2. Install SBT
    ```bash
    $echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
    $sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
    $sudo apt-get update
    $sudo apt-get install sbt
    ```
3. Configure .bashrc
4. Install py4j

Some references:
    - [Installing Apache Spark on Ubuntu 16.04](https://www.santoshsrinivas.com/installing-apache-spark-on-ubuntu-16-04/)


# Problems and Solutions
- What is the difference between JDK and JRE? ([ref][ref_1])
    - JRE: Java Runtime Environment. It is basically the Java Virtual Machine where your Java programs run on. It also includes browser plugins for Applet execution.

    - JDK: It's the full featured Software Development Kit for Java, including JRE, and the compilers and tools (like JavaDoc, and Java Debugger) to create and compile programs.
- Vim do not work well in Conemu for docker container?
    - When I start vim in a docker container in ConEmu, vim do not work well when I move cursor.
    - I solute it by update Docker for Windows from 1.13 to 1.17. The author of ConEmu said that this bug is not from ConEmu, it is because the old version docker not support it well. And the latest version of docker fix this bug.



[ref_1]: http://stackoverflow.com/questions/1906445/what-is-the-difference-between-jdk-and-jre
