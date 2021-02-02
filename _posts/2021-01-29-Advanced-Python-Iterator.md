---
title: "[Advanced Python] Iterator"
last_modified_at: 2021-01-29
categories:
  - Memo
tags:
  - Python
---

## :fallen_leaf:理解核心

需要区分**可迭代对象**和**迭代器**的概念

<p align="center">
  <img
  src="https://i.loli.net/2021/02/02/Hkf7xFOUINGKZQB.png" width="80%">
</p>

## :fallen_leaf:核心概念

- 可迭代的对象 (Iterable)：
- Object 带有`__iter__`方法时，其是可迭代对象；
- 序列都是可以迭代的；
- 有`__getitem__`方法的 Object 也是可迭代对象；
- 对可迭代对象使用 iter(可迭代对象)可以得到迭代器；
- 迭代器 (Iterator)：
- 带有**next**方法的可迭代对象是迭代器

换个角度理解，

可迭代对象需满足的条件：

1. 可以传递给 iter()函数获取到迭代器的对象

迭代器需满足的条件：

1. 可以传递给 next()函数获取到下一个元素或者抛出 StopIteration 异常的对象
2. 当传递给 iter()函数时返回的还是它自己

> Iterables:
>
> 1. Can be passed to the iter function to get an iterator for them.
> 2. There is no 2. That’s really all that’s needed to be an iterable.
>
> Iterators:
>
> 1. Can be passed to the next function which gives their next > item or raises StopIteration
> 2. Return themselves when passed to the iter function.

注：Python 的 for-loop 会先调用 iter()获取迭代器，之后调用 next()获取下一个元素

Ref: [Blog: The Iterator Protocol: How "For Loops" Work in Python](https://treyhunner.com/2016/12/python-iterator-protocol-how-for-loops-work/)

## :fallen_leaf:Generator

> @rivergold: 生成器也是迭代器，只不过是使用了`yield`或`yield from`。Python 使用生成器对惰性计算(lazy evaluation)提供了支持。

### Generator Function

生成器函数

> :bulb::bulb::bulb:@Fluent Python: 只要 Python 函数的定义体中有`yield`关键字，该函数就是生成器函数。调用生成器函数时，会返回一个生成器对象。

**_References:_**

- [Blog: Python Generator](https://lotabout.me/2017/Python-Generator/)

---

### yield from

将生成移交给子生成器

这里有一个简单的样例：

```python
def gen_x():
    yield from range(10)
```
