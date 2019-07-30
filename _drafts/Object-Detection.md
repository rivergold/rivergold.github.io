# COCO

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

**Get Annotation**

```python
image_id = image_ids[0]
annotation_id = coco.getAnnIds(imgIds=image_id)
annotation = coco.loadAnns(annotation_id)
print(annotations)
>>> [{'segmentation': [[376.97, 176.91, 398.81, 176.91, 396.38, 147.78, 447.35, 146.17, 448.16, 172.05, 448.16, 178.53, 464.34, 186.62, 464.34, 192.28, 448.97, 195.51, 447.35, 235.96, 441.69, 258.62, 454.63, 268.32, 462.72, 276.41, 471.62, 290.98, 456.25, 298.26, 439.26, 292.59, 431.98, 308.77, 442.49, 313.63, 436.02, 316.86, 429.55, 322.53, 419.84, 354.89, 402.04, 359.74, 401.24, 312.82, 370.49, 303.92, 391.53, 299.87, 391.53, 280.46, 385.06, 278.84, 381.01, 278.84, 359.17, 269.13, 373.73, 261.85, 374.54, 256.19, 378.58, 231.11, 383.44, 205.22, 385.87, 192.28, 373.73, 184.19]], 'area': 12190.44565, 'iscrowd': 0, 'image_id': 391895, 'bbox': [359.17, 146.17, 112.45, 213.57], 'category_id': 4, 'id': 151091}, ...]
```

# FPN Implementation

**:bulb:Key: Use `IntermediateLayerGetter` to get resnet intermediate return, then build a new Module**

The links show how `PyTorch torchvision` implement `resnet_fpn`:

- [class IntermediateLayerGetter](https://github.com/pytorch/vision/blob/2287c8f2dc9dcad955318cc022cabe4d53051f65/torchvision/models/_utils.py#L7)
- [class BackboneWithFPN](https://github.com/pytorch/vision/blob/2287c8f2dc9dcad955318cc022cabe4d53051f65/torchvision/models/detection/backbone_utils.py#L10)