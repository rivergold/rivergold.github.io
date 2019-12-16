# Tensor

## `contiguous()`

---

## `view()` vs `reshape()`

> @rivergold: `view()` can only work on contiguous tensor. When the tensor is not contiguous, you should use `reshape()`
> 理解：当你确定 tensor 是 contiguous 时，优先选用`view`，如果不确定时，请使用`reshape`

### [ERROR] `RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead`

**_References:_**

- :thumbsup:[stackoverflow: What's the difference between reshape and view in pytorch?](https://stackoverflow.com/a/49644300/4636081)
- [stackoverflow: PyTorch - contiguous()](https://stackoverflow.com/questions/48915810/pytorch-contiguous)

> @rivergold: `permute()` will cause tensor not contiguous anymore.

**_References:_**

- [PyTorch Forum: Call contiguous() after every permute() call?](https://discuss.pytorch.org/t/call-contiguous-after-every-permute-call/13190/2)

# nn
