# Install

Highly recommended install PyTorch follow [official website: pytorch.org](https://pytorch.org/)

:triangular_flag_on_post:**cpu**

```shell
pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

> @rivergold: 针对于 CPU 版本，强烈建议追加`+cpu`方法安装！如果仅通过`pip install torch`，Pypi 的源可能会存在 Caffe2 编译依赖 Cuda 的问题，会导致你在编译 C++文件依赖 libtorch 时出现问题

**Cuda**

```shell
pip3 install torch torchvision
```

<!--  -->
<br>

---

<br>
<!--  -->

# Common Ops

## `select`

- [PyTorch Doc](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.select)

Slices the `self` tensor along the selected dimension at the given index. This function returns a tensor with the given dimension removed.

```python
x = torch.randn(2, 3, 4)
y = x.select(0, 1)
print(y.size())
>>> torch.Size([3, 4])
# [2, 3, 4] -> [3, 4]
```

`select()` is equivalent to slicing. For example, `tensor.select(0, index)` is equivalent to `tensor[index]` and `tensor.select(2, index)` is equivalent to `tensor[:,:,index]`.

**C++**

Using `select` in C++ to achieve `tensor[:]` indexing in Python.

---

## `slice`

TODO

---

## Indexing

### [Error] `error: no match for ‘operator[]’ (operand types are ‘unsigned char*’ and ‘at::Tensor’)`

```c++
auto x = torch::rand({5});
auto x_ptr = x.data_ptr<float>();
auto idxes = torch::argsort(x, 0, true);
for (auto i = 0; i < idxes.size(0); ++i)
{
    auto idx = idxes[i];
    std::cout << x_ptr[idx] << std::endl; // Error
    std::cout << x[idx] << std::endl;     // Ok
}
```

**Solution:**

`auto idx = idxes[i];` -> `auto idx = idxes[i].item<int>();`

<!--  -->
<br>

---

<br>
<!--  -->

# Awesome Trick

## Use `tensor.contiguous()` to speed up

> @rivergold: 使用`tensor.contiguous()`会使得 PyTorch 的 tensort 变量的底层 data 数据索引是连续的，虽然`contiguous`会 copy 数据，但是该方法使用空间换时间，对于操作不是很大的张量时，有一定加速效果。

E.g. two implementation of nms:

**Slow one**

```c++
torch::Tensor nms(torch::Tensor dets, float thresh) {
  auto num_dets = dets.size(0);
  auto is_suppressed_t =
      torch::zeros(dets.size(0), dets.options().dtype(at::kByte));
  auto is_suppressed = is_suppressed_t.data_ptr<uint8_t>();
  auto sorted_idxes = torch::argsort(dets.slice(-1, 4, 5), 0, true);
  auto areas = (dets.slice(-1, 2, 3) - dets.slice(-1, 0, 1)) *
               (dets.slice(-1, 3, 4) - dets.slice(-1, 1, 2));

  for (auto i = 0; i < num_dets; ++i) {

    auto idx = sorted_idxes[i].item<int>();

    if (is_suppressed[idx] == 1)
      continue;
    for (auto other_idx = idx + 1; other_idx < num_dets; ++other_idx) {
      if (is_suppressed[other_idx] == 1)
        continue;
      auto xx1 = torch::max(dets[idx][0], dets[other_idx][0]);
      auto yy1 = torch::max(dets[idx][1], dets[other_idx][1]);
      auto xx2 = torch::min(dets[idx][2], dets[other_idx][2]);
      auto yy2 = torch::min(dets[idx][3], dets[other_idx][3]);

      auto w = std::get<0>(torch::max(xx2 - xx1, 0));
      auto h = std::get<0>(torch::max(yy2 - yy1, 0));

      auto inter_area = w * h;
      auto total_area = areas[idx] + areas[other_idx] - inter_area;
      auto iou = inter_area / total_area;

      if (iou.item<float>() > thresh)
        is_suppressed[other_idx] = 1;
    }
  }
  return dets.index(is_suppressed_t == 0);
}
```

**Fast one**

```c++
torch::Tensor nms(torch::Tensor dets, float thresh) {
  auto num_dets = dets.size(0);
  auto is_suppressed =
      torch::zeros(dets.size(0), dets.options().dtype(at::kByte));
  auto is_suppressed_ptr = is_suppressed.data_ptr<uint8_t>();

  auto scores = dets.select(1, 4).contiguous();
  auto x1 = dets.select(1, 0).contiguous();
  auto y1 = dets.select(1, 1).contiguous();
  auto x2 = dets.select(1, 2).contiguous();
  auto y2 = dets.select(1, 3).contiguous();

  auto sorted_idxes = torch::argsort(scores, 0, true);
  auto areas = (x2 - x1) * (y2 - y1);

  for (auto i = 0; i < num_dets; ++i) {

    auto idx = sorted_idxes[i].item<int>();

    if (is_suppressed_ptr[idx] == 1)
      continue;
    for (auto other_idx = idx + 1; other_idx < num_dets; ++other_idx) {
      if (is_suppressed_ptr[other_idx] == 1)
        continue;
      auto xx1 = torch::max(x1[idx], x1[other_idx]);
      auto yy1 = torch::max(y1[idx], y1[other_idx]);
      auto xx2 = torch::min(x2[idx], x2[other_idx]);
      auto yy2 = torch::min(y2[idx], y2[other_idx]);

      auto w = std::get<0>(torch::max(xx2 - xx1, 0));
      auto h = std::get<0>(torch::max(yy2 - yy1, 0));

      auto inter_area = w * h;
      auto total_area = areas[idx] + areas[other_idx] - inter_area;
      auto iou = inter_area / total_area;

      if (iou.item<float>() > thresh)
        is_suppressed_ptr[other_idx] = 1;
    }
  }
  return dets.index(is_suppressed == 0);
}
```

> @rivergold: fast one 的样例使用`select`和`contiguous`操作分别读取`x1, x2, y1, y2`

## Use `data_ptr` instead of `tensor[i].item<float>()` to speed up

> :triangular_flag_on_post: @rivergold: `tensor[i].item<float>()` is very very slow !!!

<!--  -->
<br>

---

<br>
<!--  -->

# 以下都是临时

# What is `contiguous`

**_References:_**

- :thumbsup::thumbsup::thumbsup:[知乎: PyTorch 中的 contiguous](https://zhuanlan.zhihu.com/p/64551412)

---

# View ops

Which not change or copy data, but change metadata of tensor.

- `narrow`
- `view`
- `slice`
- `expand`
- `transpose`
- `select`

**_References:_**

- [PyTorch Forum: Does select and narrow return a view or copy](https://discuss.pytorch.org/t/does-select-and-narrow-return-a-view-or-copy/289)
- [知乎: PyTorch 中的 contiguous](https://zhuanlan.zhihu.com/p/64551412)

---

# Setup PyTorch C++ Env

**_References:_**

- [PyTorch doc: INSTALLING C++ DISTRIBUTIONS OF PYTORCH](https://pytorch.org/cppdocs/installing.html)

---

# C++ Extension

## Problems & Solutions

### [macOS build error] macOS run `python setup install` occured `fatal error: 'iostream' file not foun`

**Solution:**

```shell
CFLAGS='-stdlib=libc++' python setup.py install
```

**_References:_**

- [Github pytorch/pytorch: cstddef not found when compiling C++ Extension - macOS #16805](https://github.com/pytorch/pytorch/issues/16805)
