---
title: "[Advanced Python] Concurrent"
last_modified_at: 2021-02-02
categories:
  - Memo
tags:
  - Python
---

An introduction about Python concurrent.

- Concurrent
- Threading vs Processing
- Multi threading
- Multi Processing
- Coroutines (TODO)

## :fallen_leaf:Concurrent

并发：多个任务可以同时存在，但能不能同时运行多个任务取决于 CPU；如果 CPU 是多核多线程的可以支持，单核单线程的 CPU 就不行。

### Concurrent vs Parallel

> Rob Pike 大神关于两者的阐述：“并发关乎结构，并行关乎执行”

并发: 多个任务可以同时存在
并行: 多个任务可以同时执行

代码可以写成并发的，但是如果 CPU 不支持并行，就无法并行

**_References_**

- [Rob Pike: Concurrency Is Not Parallelism](https://tech-talks.code-maven.com/concurrency-is-not-parallelism)
- [知乎: 并发与并行的区别？](https://www.zhihu.com/question/33515481/answer/105348019)

## :fallen_leaf:Threading vs Processing

### 核心概念

- 进程是程序的一次执行
- 进程是操作系统资源分配的基本单位
- 一个进程可以包括多个线程，同一进程下的多个线程共享该进程的资源
- 进程的运行不仅仅需要 CPU，还需要很多其他资源，如内存啊，显卡啊，GPS 啊，磁盘啊等等，统称为程序的执行环境，也就是程序上下文
- 进程切换：CPU 进行进程调度的时候，需要读取上下文+执行程序+保存上下文，即进程切换；进程切换的开销要比线程切换的开销大
- 线程是操作低筒调度的基本单位，CPU 只看得到线程
- 由于操作系统的调度，在 CPU 核 A 的线程也可能会被调度到核 B 上运行

**多核、超线程**

- 一个物理 CPU 可以有多个核
- 多核 CPU 的每核(每核都是一个小芯片)，在 OS 看来都是一个独立的 CPU
- 对于超线程 CPU 来说，每核 CPU 可以有多个线程(数量是两个，比如 1 核双线程，2 核 4 线程，4 核 8 线程)，每个线程都是一个虚拟的逻辑 CPU(比如 Windows 下是以逻辑处理器的名称称呼的)，而每个线程在 OS 看来也是独立的 CPU

> 超线程没有提供完全意义上的并行处理，每核 CPU 在某一时刻仍然只能运行一个进程，因为线程 1 和线程 2 是共享某核 CPU 资源的。可以简单的认为每核 CPU 在独立执行进程的能力上，有一个资源是唯一的，线程 1 获取了该资源，线程 2 就没法获取。
> 但是，线程 1 和线程 2 在很多方面上是可以并行执行的。比如可以并行取指、并行解码、并行执行指令等。所以虽然单核在同一时间只能执行一个进程，但线程 1 和线程 2 可以互相帮助，加速进程的执行。
> 并且，如果线程 1 在某一时刻获取了该核执行进程的能力，假设此刻该进程发出了 IO 请求，于是线程 1 掌握的执行进程的能力，就可以被线程 2 获取，即切换到线程 2。这是在执行线程间的切换，是非常轻量级的。(WIKI: if resources for one process are not available, then another process can continue if its resources are available)

**总结**

- 进程是分配资源的最小单位，线程是程序执行的最小单位
- 进程与进程之间的内存是独立的，线程与线程之间的内存是共享的

**_References:_**

- [掘金：多 CPU，多核，多进程，多线程，并发，并行](https://juejin.cn/post/6844903930141507592)
- [Blog: 关于 CPU 的多核和超线程技术](https://www.junmajinlong.com/os/multi_cpu/)
- [Blog: 一道面试题：说说进程和线程的区别](https://foofish.net/thread-and-process.html)
- [stackoverflow: What is the difference between a process and a thread?](https://stackoverflow.com/questions/200469/what-is-the-difference-between-a-process-and-a-thread)

## :fallen_leaf:Multi Threading

- [Python 并行编程 Cookbook](https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/index.html)

### ThreadPool

```python
import concurrent
from concurrent.futures import ThreadPoolExecutor


def job_worker(x):
    return x**2


num_jobs = 10
data = range(10)
res = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for idx, x in zip(range(num_jobs), data):
        future = executor.submit(job_worker, x)
        futures.append(future)
    # Method-1
    for future in for future in concurrent.futures.as_completed(futures):
        res.append(future.result())
    # Method-2
    # futures, _ = concurrent.futures.wait(futures)
    # for future in futures:
    #     res.append(future.result())
print(res)

>>> [0, 1, 9, 4, 16, 25, 36, 49, 64, 81]
```

注：结果是按照任务完成时间先后的顺序，可能会和原始的输出顺序不一致

## :fallen_leaf:Multi Processing

### ProcessPool

```python
import concurrent
from concurrent.futures import ProcessPoolExecutor


def job_worker(x):
    return x**2


num_jobs = 10
data = range(10)
res = []
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    for idx, x in zip(range(num_jobs), data):
        future = executor.submit(job_worker, x)
        futures.append(future)
    # Method-1
    for future in for future in concurrent.futures.as_completed(futures):
        res.append(future.result())
    # Method-2
    # futures, _ = concurrent.futures.wait(futures)
    # for future in futures:
    #     res.append(future.result())
print(res)

>>> [0, 1, 4, 16, 36, 25, 49, 9, 64, 81]
```

### 多进程中的全局变量

**Multi process will deep copy global variable by default.**

> @rivergold: Python 多进程默认不共享全局变量，而是会将全局变量拷贝到各个进程的内存中。如果需要共享数据，需要采用进程间的通信。

**_References:_**

- [CSDN: Python 多进程默认不能共享全局变量](https://blog.csdn.net/houyanhua1/article/details/78236514)
- [stackoverflow: Globals variables and Python multiprocessing [duplicate]](https://stackoverflow.com/a/11215750/4636081)
