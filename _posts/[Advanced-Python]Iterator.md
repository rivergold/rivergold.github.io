---
title: "[Advanced Python] Iterator"
last_modified_at: 2021-01-29
categories:
  - Memo
tags:
  - Python
---

## 理解核心

需要区分**可迭代对象**和**迭代器**的概念

## 核心概念：

- 可迭代的对象 (Iterable)：
- Object带有`__iter__`方法时，其是可迭代对象；
- 序列都是可以迭代的；
- 有`__getitem__`方法的Object也是可迭代对象；
- 对可迭代对象使用iter(可迭代对象)可以得到迭代器；
- 迭代器 (Iterator)：
- 带有__next__方法的可迭代对象是迭代器

换个角度理解，

可迭代对象需满足的条件：
1. 可以传递给iter()函数获取到迭代器的对象

迭代器需满足的条件：

1. 可以传递给next()函数获取到下一个元素或者抛出StopIteration异常的对象
2. 当传递给iter()函数时返回的还是它自己

> Iterables:
> 1. Can be passed to the iter function to get an iterator for them.
> 2. There is no 2. That’s really all that’s needed to be an iterable.
>
> Iterators:
> 1. Can be passed to the next function which gives their next > item or raises StopIteration
> 2. Return themselves when passed to the iter function.

注：Python的for-loop会先调用iter()获取迭代器，之后调用next()获取下一个元素

Ref: [Blog: The Iterator Protocol: How "For Loops" Work in Python](https://treyhunner.com/2016/12/python-iterator-protocol-how-for-loops-work/)
