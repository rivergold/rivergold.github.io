---
title: "PyTorch and Numpy Common Ops Collection"
last_modified_at: 2020-02-22
categories:
  - Memo
tags:
  - PyTorch
  - Python
  - Numpy
---

A collation of common ops in PyTorch and Numpy.

## :fallen_leaf:Basics

Most operations has `variable.func` and `package.func`.

**Numpy**

- `np.func`
- `np.ndarray.func`

**PyTorch**

- `torch.func`
- `torch.Tensor.func`

Note: if example code use `package.func`, it means that there is no `variable.func` for this operation.

### View

#### Numpy check is view or not

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
# b is view of a
b = a.ravel()
print(b.base is a)
>>> True
print(np.may_share_memory(a, b))
>>> True
print(b.flags['OWNDATA'])
>>> False
```

**_Ref:_** [stackoverflow: How can I tell if NumPy creates a view or a copy?](https://stackoverflow.com/questions/11524664/how-can-i-tell-if-numpy-creates-a-view-or-a-copy)

### Broadcasting

Broadcasting only process when the operation is element-wise operation.

**:star2:The Broadcasting Rule:**

**In order to broadcast, the size of the trailing axes for both arrays in an operation must either be the same size or one of them must be one.**

@rivergold: **将参与计算操作的两个矩阵的 size 进行右对齐，对应位置的维度要么相等，要么其中一个为 1**

E.G.

```shell
Image	(3d array)	256 x	256 x	3
Scale	(1d array)	 	 	3
Result	(3d array)	256 x	256 x	3
```

```shell
A	(4d array)	8 x	1 x	6 x	1
B	(3d array)	 	7 x	1 x	5
Result	(4d array)	8 x	7 x	6 x	5
```

**_Ref:_** [Numpy doc: Array Broadcasting in Numpy](https://docs.scipy.org/doc/numpy/user/theory.broadcasting.html#array-broadcasting-in-numpy)

## :fallen_leaf:Index

**Numpy**

```python
x = np.random.randn(2, 3, 2)
# Dimension decrease
y = x[:, :, 0]
print(y.shape)
>>> (2, 3)
# Dimension not change
y = x[:, :, 0:1]
print(y.shape)
>>> (2, 3, 1)`
```

**PyTorch**

```python
x = torch.randn(2, 3, 2)
# Dimension decrease
y = x[:, :, 0]
print(y.size())
>>> torch.Size([2, 3])
# Dimension not change
y = x[:, :, 0:1]
print(y.size())
>>> torch.Size([2, 3, 1])
```

### :triangular_flag_on_post:Index with array/Tensor

```
0, 1, 2
3, 4, 5  -> how to get [2, 4, 8]
6, 7, 8
```

**Numpy**

```python
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# Index each row specific place element
idx = np.array([2, 1, 2])
y = x[range(x.shape[0]), idx]
print(y)
>>> array([2, 4, 8])
# Can not do via [:, idx], because it will select idx on each row
y = x[:, idx]
print(y)
>>> array([[2, 1, 2],
           [5, 4, 5],
           [8, 7, 8]])
```

**PyTorch**

```python
x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# Index each row specific place element
idx = torch.tensor([2, 1, 2])
y = x[range(x.size(0)), idx]
print(y)
>>> tensor([2, 4, 8])
# Can not do via [:, idx], because it will select idx on each row
y = x[:, idx]
print(y)
>>> tensor([[2, 1, 2],
            [5, 4, 5],
            [8, 7, 8]])

```

## :fallen_leaf:Flatten

Convert $n \times w$ to 1-D vector.

**Numpy**

```python
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x.flatten()
print(x)
>>> array([1, 2, 3, 4, 5, 6])
```

Note: result of `np.ravel()` is same as `np.flatten`, but `np.ravel`'s return is **view**, which means if you change `y`, will also change `x`.

**_Ref:_** [stackoverflow: What is the difference between flatten and ravel functions in numpy?](https://stackoverflow.com/a/28930580/4636081)

**PyTorch**

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = x.flatten()
>>> tensor([1, 2, 3, 4, 5, 6])
```

Note: `torch.Tensor.view(-1)` has the same effect with `np.ndarray.ravel()`.

## :fallen_leaf:Squeeze

Remove dimensions which size is 1.

**Numpy**

```python
x = np.random.randn(1, 2, 1)
print(x)
>>> array([[[ 0.        ],
            [-0.52306498]]])
# Note: y is view of x
y = x.squeeze()
print(y, y.shape)
>>> [ 0.         -0.52306498] (2,)
```

**PyTorch**

```python
x = torch.randn(1, 2, 1)
print(x)
>>> tensor([[[ 0.0000],
             [-0.7718]]])

# Note: y is view of x
y = x.squeeze(x)
print(y, y.size())
>>> tensor([ 0.0000, -0.7718]) torch.Size([2])
```

## :fallen_leaf:Unsqueeze

Insert new dimension before a axis.

**Numpy**

```python
x = np.array([1, 2, 3])
# Note: y is view of x
# Method 1
y = np.expand_dims(x, 0)
# Method 2
y = x[np.newaxis, ...]
print(y, y.shape)
>>> [[1 2 3]] (1, 3)
```

**PyTorch**

```python
x = torch.tensor([1, 2, 3])
# Note: y is view of x
y = x.unsqueeze(0)
print(y, y.size())
>>> tensor([[1, 2, 3]]) torch.Size([1, 3])
```

## :fallen_leaf:Swap Axis

**Numpy**

```python
x = np.random.randn(2, 3, 4)
# Note: y is view of x
# Method 1
y = x.transpose(1, 2, 0)
# Method 2
y = np.transpose(x, (1, 2, 0))
print(y.shape)
>>> (3, 4, 2)
```

**PyTorch**

```python
x = torch.randn(2, 3, 4)
# Note: y is view of x
y = x.permute(1, 2, 0)
print(y.size())
>>> torch.Size([3, 4, 2])
```

## :fallen_leaf:Clip Value

**Numpy**

```python
x = numpy.array([-2. 3])
# Method 1
y = x.clip(min=0, max=5)
# Method 2
y = np.clip(x, 0, 5)
print(y)
>>> array([0, 3])
```

**PyTorch**

```python
x = torch.tensor([-2, 3])
# Method 1
y = x.clamp(min=0, max=5)
# Method 2
y = torch.clamp(x, min=0, max=5)
print(y)
>>> tensor([0., 3.])
```

## :fallen_leaf:Max

### Single Array or Tensor

**Numpy**

**Only get max value of one `np.ndarray`, without index. `numpy` cannot get max value and index together.**

```python
x = np.array([[-1, 5], [6, 0]])
# axis=1
# 理解：按列比较（每列之间进行比较，找到最大的）
y = np.max(x, axis=1)
print(y)
>>> array([6, 5])
# axis=0
# 理解：按行比较（每行之间进行比较，找到最大的）
y = np.max(x, axis=0)
print(y)
>>> array([5, 6])
```

:triangular_flag_on_post:**PyTorch**

**Get max value of one `Tensor` and max value index.**

```python
x = torch.tensor([[-1, 5], [6, 0]])
# axis=1
# Method 1
value, idx = torch.max(x, dim=1)
# Method 2
value, idx = x.max(dim=1)
print(value)
>>> tensor([5, 6])
print(idx)
>>> tensor([1, 0])
# axis=0
value, idx = torch.max(x, dim=0)
print(value)
>>> tensor([6, 5])
print(idx)
>>> tensor([1, 0])
```

### Two Array or Tensor

**Numpy**

```python
x1 = np.random.randn(3, 2)
x2 = np.random.randn(3, 2)
y = np.maximum(x1, x2)
print(y.shape)
>>> (3, 2)
```

**PyTorch**

```python
x1 = torch.randn(3, 2)
x2 = torch.randn(3, 2)
y = torch.max(x1, x2)
print(y.size())
>>> torch.Size([3, 2])
```

## :fallen_leaf:Argmax

Returns the indices of the maximum values of a tensor across a dimension.

**Numpy**

```python
x = np.array([[-1, 5], [6, 0]])
idx = np.argmax(x, axis=1)
print(idx)
>>> array([1, 0])
```

**PyTorch**

```python
x = torch.tensor([[-1, 5], [6, 0]])
idx = torch.argmax(x, axis=1)
print(idx)
>>> tensor([1, 0])
```

## :fallen_leaf:Where

**Numpy**

```python
# np.where
x = np.array([[1, 2], [3, 4]])
idx_0, idx_1 = np.where(x>1)
print(idx_0)
>>> array([0, 1, 1]
print(idx_1)
>>> array([1, 0, 1])
# ------------------------------
# np.argwhere
idx = np.argwhere(x>1)
print(idx)
array([[0, 1],
       [1, 0],
       [1, 1]])
```

**PyTorch**

```python
x = torch.tensor([[1, 2], [3, 4]])
idx_0, idx_1 = torch.where(x>1)
print(idx_0)
>>> tensor([0, 1, 1])
print(idx_1)
>>> tensor([1, 0, 1])
```

## :fallen_leaf:Argsort

**numpy**

```python
x = np.random.randn(10, 5)
# Ascending (default)
idxes = np.argsort(x)
# Descending
idxes = np.argsort(x)[::-1]
```

**PyTorch**

```python
torch.argsort(input, dim=-1, descending=False, out=None) → LongTensor
```

```python
x = torch.randn(10, 5)
# Ascending (default)
idxes = torch.argsort(x)
# Descending
idxes = torch.argsort(x[:, -1], descending=True)
```

## :fallen_leaf:Concate

- [ ] TBD

**Numpy**

**PyTorch**

### PyTorch `cat` vs `stack`

**_Ref:_** [stackoverflow: What's the difference between torch.stack() and torch.cat() functions?](https://stackoverflow.com/questions/54307225/whats-the-difference-between-torch-stack-and-torch-cat-functions/54307331)
