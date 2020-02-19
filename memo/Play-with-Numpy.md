# Create

## Create an empty ndarray

```python
a = np.zeros((0, 4))
>>> array([], shape=(0, 4), dtype=float64)
```

**_Ref:_** [yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet/blob/1135e18b835481b18fd0d4e1613c87afc2bc7d46/dataloader.py#L86)

<!--  -->
<br>

---

<br>
<!--  -->

# Index

Ref [Numpy Doc: Indexing routines](https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.indexing.html)

## Condition indexing -- `logic`

Better use `np.logical_and`,

```python
a = np.random.randn(5, 5, 3)
a[np.logical_and(a[:,:,0]>0, a[:,:,1]>0)]
```

Ref [stackoverflow: Difference between 'and' (boolean) vs. '&' (bitwise) in python. Why difference in behavior with lists vs numpy arrays?](https://stackoverflow.com/q/22646463/4636081)

Or you can using `() & ()`. Because there is no logical value of `ndarray`, so you need to use `&`.

```python
a = np.random.randn(5, 5, 3)
a[(a[:,:,0]>0) & a[:,:,1]>1].shape
>>> (3, 3)
```

Ref [Python Data Science Handbook: Comparisons, Masks, and Boolean Logic](https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html#Boolean-operators)

---

## Condition indexing -- `np.argwhere`

```python
a = np.array([1,2,3,4])
b = np.argwhere(a>2)
>>> array([[2],
          [3]])
```

**_Ref:_** [Numpy Doc: numpy.argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)

---

## `np.nonzero(a)`

Return the indices of the elements that are non-zero.

```python
yy, xx = np.nonzero(img==255)
```

**_References:_**

- [Numpy Doc: numpy.nonzero](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.nonzero.html)

---

## `numpy.unique`

Find the unique elements of an array.

**Calculate element count in a ndarray**

```python
a = np.random.randn(4, 4)
unique, counts = numpy.unique(a, return_counts=True)
```

Ref [stackoverflow: How to count the occurrence of certain item in an ndarray in Python?](https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python)

<!--  -->
<br>

---

<br>
<!--  -->

# Shape

## `np.squeeze(a, axis=None)`

Remove single-dimensional entries from the shape of an array.

```python
x = np.array([[[0], [1], [2]]])
x.shape
>>> (1, 3, 1)
np.squeeze(x).shape
>>> (3,)
```

Ref [Numpy doc: numpy.squeeze](https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html)

<!--  -->
<br>

---

<br>
<!--  -->

# Random

## `np.random.RandomState`

```python
def microstructure(l=256):
    n = 5
    x, y = np.ogrid[0:l, 0:l]  #生成网络
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)  #随机数种子
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l/(4.*n)) #高斯滤波
    return mask > mask.mean()
```

**_References:_**

- [Numpy Doc: numpy.random.RandomState](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html)
- [denny 的学习专栏: python 数字图像处理（18）：高级形态学处理](https://www.cnblogs.com/denny402/p/5166258.html)

<!--  -->
<br>

---

<br>
<!--  -->

# Problems & Solutions

## :triangular_flag_on_post:[Be Careful] When use `ndarray.astype(np.uint8)`, the value range is [0, 255], it will be overflow!

E.g

```python
a = np.array([256])
print(a)
>>> [256]
a = a.astype(np.uint8)
print(a)
>>> [0]
```
