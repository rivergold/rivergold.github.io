# :fallen_leaf: New

## Python 3.8

***References:***

- [Python开发者: Python 3.8 新功能大揭秘](https://mp.weixin.qq.com/s?__biz=MzA4MjEyNTA5Mw==&mid=2652569675&idx=1&sn=928715a62ad52aebd66b1076c6b7b4ff&chksm=84652801b312a1170c2beb98b25431353e5dec43c346390f2bf2ce91256c0313d69c98a15c3a&mpshare=1&scene=1&srcid=#rd)

# :fallen_leaf:Awesome Tricks

## Make Class like a `dict`

Can use `[]` or `get` to get attribute in the class.

**Good example: [Github open-mmlab: mmcv/mmcv/utils/config.py](https://github.com/open-mmlab/mmcv/blob/9097957434da1308739c03aff53ac147f1500c56/mmcv/utils/config.py#L142)**

<!--  -->
<br>

***
<!--  -->

## Set random seed

```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

***Ref:*** [Github open-mmlab/mmdetection: mmdet/apis/env.py](https://github.com/open-mmlab/mmdetection/blob/58c415a069ceab37bab76c47da72824f4181cff6/mmdet/apis/env.py#L53)