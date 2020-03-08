---
title: "[Advanced PyTorch] Dataset and DataLoader"
last_modified_at: 2020-03-05
categories:
  - Memo
tags:
  - PyTorch
  - Tool
  - Memo
---

Understand usage of PyTorch Dataset and DataLoader.

## :fallen_leaf:Dataset

[PyTorch Doc: TORCH.UTILS.DATA](https://pytorch.org/docs/stable/data.html)

### Map-Style Dataset

> @rivergold: 我们常用的`Dataset`属于 Map-Style Dataset，需要用户实现`__getitem__()`和`__len__()`，之后通过`idx`获取 sample

### Iterable-Style Dataset

> @rivergold: `IterableDataset`本质上是一个迭代器，用户需要实现`__iter__()`，之后通过`iter(dataset)`获取 sample

## :fallen_leaf:DataLoader

[PyTorch Doc: TORCH.UTILS.DATA](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

### sampler

> @rivergold: 本质上，sampler 是一个迭代器（也可以是一个生成器），用来产生 DataLoader 所需要的 index

**_Ref_** :thumbsup: :thumbsup: :thumbsup::triangular_flag_on_post::triangular_flag_on_post::triangular_flag_on_post:[Detectron2: TrainingSampler](https://github.com/facebookresearch/detectron2/blob/ef096f9b2fbedca335f7476b715426594673f463/detectron2/data/samplers/distributed_sampler.py#L12)

#### sampler conflicts with `shuffle=True`

When set `sampler=xxx`, must use `shuffle=False`. If not, will occur error `ValueError: sampler option is mutually exclusive with shuffle`

**_Ref_** [Blog: ValueError sampler is mutually exclusive with shuffle](http://www.iterate.site/post/01-%E6%8E%A2%E7%B4%A2/04-%E6%A1%86%E6%9E%B6%E4%BD%BF%E7%94%A8/11-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6/11-pytorch/valueerror-sampler-is-mutually-exclusive-with-shuffle/)

### collate_fn

> @PyTorch: You can use your own collate_fn to process the list of samples to form a batch.

You can define your own `collate_fn` to realize a dataloader with variable-size input.

**_References:_**

- :thumbsup:[PyTorch Forum: How to use collate_fn()](https://discuss.pytorch.org/t/how-to-use-collate-fn/27181/2?u=rivergold)
- :thumbsup:[PyTorch Forum: How to create a dataloader with variable-size input](https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2)

### pin_memory

**_References:_**

- [PyTorch Forum: What is the disadvantage of using pin_memory?](https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702)
- [PyTorch Forum: When to set pin_memory to true?](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2)

## :fallen_leaf:Tricks

### :bulb::triangular_flag_on_post:Generate infinite data stream for training

- torch.utils.data.Dataset
- torch.utils.data.Sampler
- Python generator

> @rivergold: 将 Sampler 和 Python 的生成器结合在一起，一直产生在样本总数以内的 index，以阻止 DataLoader 抛出 StopIteration 的异常

**_Ref_** :thumbsup::thumbsup::thumbsup:[Detectron2: TrainingSampler](https://github.com/facebookresearch/detectron2/blob/ef096f9b2fbedca335f7476b715426594673f463/detectron2/data/samplers/distributed_sampler.py#L12)
