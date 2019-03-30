# Functions

## Index

Ref [Numpy Doc: Indexing routines](https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.indexing.html)

### `np.nonzero(a)`

Return the indices of the elements that are non-zero.

```python
yy, xx = np.nonzero(img==255)
```

***References:***

- [Numpy Doc: numpy.nonzero](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.nonzero.html)

<!--  -->
<br>

***

<br>
<!--  -->

## Shape

### `np.squeeze(a, axis=None)`

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

***

<br>
<!--  -->

## Random

### `np.random.RandomState`

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

***References:***

- [Numpy Doc: numpy.random.RandomState](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html)
- [denny的学习专栏: python数字图像处理（18）：高级形态学处理](https://www.cnblogs.com/denny402/p/5166258.html)

<!--  -->
<br>

***

<br>
<!--  -->