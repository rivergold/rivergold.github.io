# Awesome Tricks

## Make Class like a `dict`

Can use `[]` or `get` to get attribute in the class.

**Good example: [Github open-mmlab: mmcv/mmcv/utils/config.py](https://github.com/open-mmlab/mmcv/blob/9097957434da1308739c03aff53ac147f1500c56/mmcv/utils/config.py#L142)**

## Set random seed 

```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

***Ref:*** [Github open-mmlab/mmdetection: mmdet/apis/env.py](https://github.com/open-mmlab/mmdetection/blob/58c415a069ceab37bab76c47da72824f4181cff6/mmdet/apis/env.py#L53)