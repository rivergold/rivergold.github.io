# Awesome Tools

# imgaug

- [Github Home: aleju/imgaug](https://github.com/aleju/imgaug)

## Problems & Solutions

### `seq.to_deterministic()`的作用是什么

```python
# Make our sequence deterministic.
# We can now apply it to the image and then to the keypoints and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the
# exactly same augmentations for every batch!
seq_det = seq.to_deterministic()
```

Ref [imgaug doc: Examples: Keypoints](https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html)

***如果没有在loop中调用该函数，你所得到的每次结果都是一样的。所以在loop中必须调用该函数。***

***我的理解: `seq.to_deterministic()`该函数临时确定了一个random值，提供给seq中的取值***

***References:***

- [Github aleju/imgaug: What is the functionality of "seq.to_deterministic()"? #26](https://github.com/aleju/imgaug/issues/26#issuecomment-290923367)
- [Github aleju/imgaug: about the use of seq.to_deterministic() #162](https://github.com/aleju/imgaug/issues/162)