---
title: "Advanced PyTorch"
categories:
  - memo
tags:
  - pytorch
  - tool
  - memo
---

Some advanced usage of PyTorch.

## :fallen_leaf:Dynamic Graph

> @rivergold: 在 forward 中构建图，在 backward 中析构图

## :fallen_leaf:DataLoader

TODO

## :fallen_leaf:Data Parallel

:thumbsup:[PyTorch Tutorial: MULTI-GPU EXAMPLES](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)

> @pytorch: Data Parallelism is when we split the mini-batch of samples into multiple smaller mini-batches and run the computation for each of the smaller mini-batches in parallel.

数据并行：不同的机器有同一个模型的多个副本，每个机器分配到不同的数据，然后将所有机器的计算结果按照某种方式合并。

E.g.

```python
# Build your model
# model = xxx
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

TODO: how weights update?

**_Ref_** [PyTorch Tutorial: DATA PARALLELISM](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)

## :fallen_leaf:Distributed Data Parallel

关于 Distributed Data Parallel 和上一节的 Data Parallel 的区别，可以阅读[Distributed data parallel training in Pytorch](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)。

**_Ref_** :thumbsup:[知乎: Pytorch 中的 Distributed Data Parallel 与混合精度训练（Apex）](https://zhuanlan.zhihu.com/p/105755472?utm_source=ZHShareTargetIDMore&utm_medium=social&utm_oi=37839882420224)
