# Generative Image Inpainting with Contextual Attention

```python
h, w, _ = image.shape
grid = 8
image = image[:h//grid*grid, :w//grid*grid, :]
mask = mask[:h//grid*grid, :w//grid*grid, :]
print('Shape of image: {}'.format(image.shape))
```

`//` 计算得到`商`

***References:***

- [stackoverflow: What do these operators mean (** , ^ , %, //)? [closed]](https://stackoverflow.com/questions/15193927/what-do-these-operators-mean)

---

```python
image = np.expand_dims(image, 0)
mask = np.expand_dims(mask, 0)
input_image = np.concatenate([image, mask], axis=2)
```

`numpy.expand_dims(a, axis)`: add a new axis before `axis`

***References:***

- [Scipy.org Docs: numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html)

---

Tensorflow dimension order: (num_samples, h, w, channel)

***References:***

- [stackoverflow: Why tensorflow uses channel-last ordering instead of row-major?](https://stackoverflow.com/questions/44774234/why-tensorflow-uses-channel-last-ordering-instead-of-row-major)

$inf{}$: 最大下界

- [wiki: 最大下界](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%B8%8B%E7%95%8C)

论文中的deconv的实现为：将图片resize为输入的两倍，之后进行卷积

<br>


***

<br>

# Image Inpainting for Irregular Holes Using Partial Convolutions

- [Github naoto0804/pytorch-inpainting-with-partial-conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv)

## Key

- Model input: img_with_hole, mask, range is `[0, 1]`, then normailze wih `transforms.Normalize(mean=opt.MEAN, std=opt.STD)`
    - `mask=1`: image
    - `mask=0`: hole
- output: need to be `unnormalize` and then `* 255`