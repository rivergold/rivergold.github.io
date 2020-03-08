# 图形学

## 连通域

```python
from skimage import morphology, color
# img: np.ndarray
# labels: np.ndarray. Labeled array, where all connected regions are assigned the same integer value.
labels = morphology.label(img, connectivity=2) # skimage.measure.label is same
```

**_References:_**

- [scikit-image Doc Module: morphology](http://scikit-image.org/docs/dev/api/skimage.morphology.html#label)
- [scikit-image Doc Module: measure](https://scikit-image.org/docs/dev/api/skimage.measure.html#label)
- [denny 的学习专栏: python 数字图像处理（18）：高级形态学处理](https://www.cnblogs.com/denny402/p/5166258.html)

<!--  -->
<br>

---

<br>
<!--  -->

# 基础

## 亮度、对比度、饱和度、锐化、分辨率

**_Ref:_** [知乎: 【数字图像处理系列二】亮度、对比度、饱和度、锐化、分辨率](https://zhuanlan.zhihu.com/p/44813768)

## 色相、亮度、饱和度

- 色相: 颜色的主色调
- 亮度: 颜色的明暗
- 饱和度: 颜色的鲜艳成都

**_Ref:_** [Hanks Home: 一张图让开发人员理解色相、亮度、饱和度](https://hanks.pub/2016/03/26/color-board/)

## 对比度

通俗的讲，就是亮暗的对比程度
对比度通常表现了图像画质的清晰程度

**_Ref:_** [PDF 图像灰度增强](http://read.pudn.com/downloads87/ebook/335162/%E7%AC%AC3%E7%AB%A0%20%E7%81%B0%E5%BA%A6%E7%BA%A7%E5%8F%98%E6%8D%A2.ppt)

## 颜色空间

### HSV

**_Ref:_** [HSL 和 HSV 色彩空间](https://zh.wikipedia.org/wiki/HSL%E5%92%8CHSV%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4)

<!--  -->
<br>

---

<br>
<!--  -->

# 算法

## Alpha Blending

Merge two image with alpha channel together.

```python
# TBD
```

**_Ref:_** [wiki: Alpha compositing](https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending)

**_References:_**

- [stackoverflow: How does photoshop blend two images together?](https://stackoverflow.com/questions/5919663/how-does-photoshop-blend-two-images-together)

<!--  -->
<br>

---

<br>
<!--  -->

# Tools

## For Python, OpenCV, skimage, scipy

Here is a [reference](https://mmas.github.io/python-image-processing-libraries-performance-opencv-scipy-scikit-image) for performance compare.

Maybe, you'd better chose OpenCV first.

<!--  -->
<br>

---

<!--  -->

# ImageMagick

- [ImageMagick Doc](http://www.fifi.org/doc/imagemagick/www/ImageMagick.html)

- [图像处理 - ImageMagick 简单介绍与案例](https://aotu.io/notes/2018/06/06/ImageMagick_intro/index.html)

## Convert png into PSD

```bash
convert xc:none[750x210+0+0] logo.png title.png subtitle.png test.psd
```

**Note: It only work on CentOS (ImageMagick 6.7.8-9). On Ubuntu-18.04 (ImageMagick 6.9.7-4), cannot open the psd file.**

- [ ] TBD

**_References:_**

- [ImageMagick Forum: Montage and save in layered psd](http://www.imagemagick.org/discourse-server/viewtopic.php?t=31707)

- [ImageMagick Forum: Create PSD with IM with locked layers](http://www.imagemagick.org/discourse-server/viewtopic.php?f=1&t=27740)

- [ImageMagick Forum: Create psd file with a single transparent layer](http://www.imagemagick.org/discourse-server/viewtopic.php?t=33186&start=30)

### Put image position

```bash
convert xc:none[900x900+0+0] -page 10x10+0+0 raw_logo.png -page +100+100 logo.png -depth 16 test.psd
```

---

## Convert psd to image

```bash
# Convert each layer into an image
convert <psd_file_path> <out_image_path>
# Convert all layers into only one image
convert -flatten <psd_file_path> <out_image_path>
```

**_References:_**

- [StackExchange: Any way to save all \*.psd layers separately](https://superuser.com/questions/360434/any-way-to-save-all-psd-layers-separately)

- [stackoverflow: Convert PSD with custom channels to JPG](https://stackoverflow.com/questions/10874691/convert-psd-with-custom-channels-to-jpg)

---

## Convert Color

### Convert `sRGB` into `CMYK`

:triangular_flag_on_post:CMYK 所能表示的色彩范围比 sRGB 要小，因此从 sRGB 转换到 CMYK 会存在颜色的变化。所以在进行转换时，需要找到最合适的方法，使得颜色损失最少。

**转化的方法基于 ImageMagick 的`convert`**

转化时需要配置文件 **color profiles**:

- sRBG 图片所用的 color profile 为: `sRGB Color Space Profile.icm`
- photoshop 对 CMYK 所使用的 profile 为: `JapanColor2001Coated.icc`

配置文件在 windows 系统下的`C:\Windows\System32\spool\drivers\color`，该配置文件经`ImageMagick`测试可以在 Linux 系统上使用。

有些图片会有 profile，有些图片没有 profile，可以使用`identify -verbose <image_path>`来获取图片的所用信息，其中 profile 配置文件的字段为`Profile`，如果没有的话，就表示该图片没有配置文件的信息。

**with color profile**

- [ ] TBD

**without color profile**

```shell
convert <sRGB_image_path> +profile icm -profile <sRGB Color Space Profile.icm file path> -profile <JapanColor2001Coated.icc file path> <output_tif_path>
```

**_References:_**

- :thumbsup:[ImageMagick Doc: Common Image Formats - Image Profiles](http://www.imagemagick.org/Usage/formats/#profiles)

- [ImageMagick Forum: convert RGB to CMYK colorspace](https://www.imagemagick.org/discourse-server/viewtopic.php?t=16572)

**What is color profiles?**

目前的理解是，定义了 color 的表示方式以及转换关系

### Convert `CMYK` into `sRGB`

```shell
convert -colorspace sRGB cmyk.tif sRGB.tif
```

**_Ref:_** [ImageMgick: Convert CMYK to RGB jpgs - strange colors](https://www.imagemagick.org/discourse-server/viewtopic.php?t=13398)

---

## Problems & Solutions

### On Ubuntu use ImageMagick to generate psd, occur 'Error compression version, cannot open'

**Solution:** When use `convert`, need to set compression version as `RLE`

```bash
convert <image path> -compress RLE <psd path>
```

**_Ref:_** [Image Magick: convert](https://imagemagick.org/script/convert.php)

- [ImageMagick: How to write multi-layer Photoshop PSD file?](http://www.imagemagick.org/discourse-server/viewtopic.php?t=32083)

### convert: delegate library support not built-in

```shell
convert: delegate library support not built-in `/root/tmp/data/tmp/20191029-094250-719826.png' (LCMS) @ warning/profile.c/ProfileImage/565.
```

TODO:

**_References:_**

- [ImageMagick: delegate library support not built-in](https://www.imagemagick.org/discourse-server/viewtopic.php?t=21070)

<!--  -->
<br>

---

<br>
<!--  -->

# 图片写文字

## Python `wand`

- [ ] TBD

### 文字间距

```python
with Drawing() as draw:
    draw.font = font_path.as_posix()
    draw.font_size = font_size
    draw.fill_color = Color(font_color)
    draw.text_interword_spacing = 100
```

只有当文字（原始字符串）之间有空格时， `text_interword_spacing`才会有作用

# Convet HEX color into RGB color

```python
hex_color = 'fcfad5'
rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
```

**_References:_**

- [stackoverflow: Converting Hex to RGB value in Python](https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python)

<!--  -->
<br>

---

<br>
<!--  -->

# Python Read gif

```python
import imageio
import cv2

def read_gif(in_gif_path):
    raw_gif_images = imageio.mimread(in_gif_path)
    gif_images = []
    for gif_image in raw_gif_images:
        gif_image = cv2.cvtColor(gif_image, cv2.COLOR_RGB2BGR)
```

**_References:_**

- [stackoverflow: How to read gif from url using opencv (python)](https://stackoverflow.com/questions/48163539/how-to-read-gif-from-url-using-opencv-python)

<!--  -->
<br>

---

<br>
<!--  -->

# 羽化操作

羽化 = 边缘模糊

1. 模糊整张图片
2. 找到边缘
3. 将模糊的边缘和没有模糊的内部结合起来

TODO:

**_References:_**

- :thumbsup:[stackoverflow: How to blur/ feather the edges of an object in an image using Opencv](https://stackoverflow.com/questions/55066764/how-to-blur-feather-the-edges-of-an-object-in-an-image-using-opencv)