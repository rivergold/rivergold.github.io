# ResNet

## Blogs

- [CSDN: ResNet 解析](https://blog.csdn.net/lanran2/article/details/79057994)

<!--  -->
<br>

---

<br>
<!--  -->

# FPN

## Blogs

- [知乎: 目标检测算法综述之 FPN 优化篇](https://zhuanlan.zhihu.com/p/63047557)

## FPN Implementation

**:bulb:Key: Use `IntermediateLayerGetter` to get resnet intermediate return, then build a new Module**

The links show how `PyTorch torchvision` implement `resnet_fpn`:

- [class IntermediateLayerGetter](https://github.com/pytorch/vision/blob/2287c8f2dc9dcad955318cc022cabe4d53051f65/torchvision/models/_utils.py#L7)
- [class BackboneWithFPN](https://github.com/pytorch/vision/blob/2287c8f2dc9dcad955318cc022cabe4d53051f65/torchvision/models/detection/backbone_utils.py#L10)

<!--  -->
<br>

---

<br>
<!--  -->

# RetinaNet

## Question and Answer

### RetinaNet use FPN P3~P7 level feature map, where P6 and P7 calculate from?

ResNet has C1~C5 stages. P3~P5 level feature map calculate from C3~C5. P6 calculate from C5 and P7 calculate from P6.

```python
P6 = Conv3x3 + stride=2 (C5)
P7 = Relu + Conv3x3 + stride=2 (P6)
```

Paper note-2 explain this.

<!--  -->
<br>

---

<br>
<!--  -->

# Anchor

## Blogs

- [知乎: faster rcnn 中 rpn 的 anchor，sliding windows，proposals？](https://www.zhihu.com/question/42205480/answer/525212289)

- [物体检测中的 anchor](https://mingming97.github.io/2019/03/26/anchor%20in%20object%20detection/)

- [Github rbgirshick/py-faster-rcnn: The problems about anchors ? #112](https://github.com/rbgirshick/py-faster-rcnn/issues/112)

<!--  -->
<br>

---

<br>
<!--  -->

# COCO

## :triangular_flag_on_post:Information

There are 91 classes in the coco paper. But in 2014 and 2017 dataset, there are 80 classes in it.

**_References:_**

- [Amikelive | Technology Blog: What Object Categories / Labels Are In COCO Dataset?](https://tech.amikelive.com/)

## Load COCO dataset

1. Init coco with annotation file
2. Get image_ids and categories
3. Use image_id to get image infomation and annotation

```python
from pycocotools.coco import COCO
```

It's better to use ImageId as index to get each image annotations.

**Init COCO**

```python
annotation_path = <your_annotation_json_file_path>
coco = coco = COCO(annotation_path.as_posix())
```

**Get ImageIds**

```python
image_ids = coco.getImgIds()
```

**Get Categories**

```python
categories = coco.loadCats(coco.getCatIds())
print(categories)
>>> [{'supercategory': 'person', 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, ...]
```

**Get Image**

```python
img_info = coco.loadImgs(img_id)[0]
img_name = img_info['file_name']
```

**Get Annotation**

```python
image_id = image_ids[0]
annotation_id = coco.getAnnIds(imgIds=image_id)
annotation = coco.loadAnns(annotation_id)
print(annotations)
>>> [{'segmentation': [[376.97, 176.91, 398.81, 176.91, 396.38, 147.78, 447.35, 146.17, 448.16, 172.05, 448.16, 178.53, 464.34, 186.62, 464.34, 192.28, 448.97, 195.51, 447.35, 235.96, 441.69, 258.62, 454.63, 268.32, 462.72, 276.41, 471.62, 290.98, 456.25, 298.26, 439.26, 292.59, 431.98, 308.77, 442.49, 313.63, 436.02, 316.86, 429.55, 322.53, 419.84, 354.89, 402.04, 359.74, 401.24, 312.82, 370.49, 303.92, 391.53, 299.87, 391.53, 280.46, 385.06, 278.84, 381.01, 278.84, 359.17, 269.13, 373.73, 261.85, 374.54, 256.19, 378.58, 231.11, 383.44, 205.22, 385.87, 192.28, 373.73, 184.19]], 'area': 12190.44565, 'iscrowd': 0, 'image_id': 391895, 'bbox': [359.17, 146.17, 112.45, 213.57], 'category_id': 4, 'id': 151091}, ...]
```

# Image Size

**_References:_**

- [fizyr/keras-retinanet: allow training with larger batch size #25](https://github.com/fizyr/keras-retinanet/issues/25)
