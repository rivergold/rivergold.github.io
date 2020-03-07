---
title: "[Advanced-PyTorch] Tricks"
last_modified_at: 2020-03-04
categories:
  - Memo
tags:
  - PyTorch
  - Memo
---

## InterLayerGetter

Get some layer output from a model, then pass these output feature to another model. The typical usage is in FPN to get C_x feature.

> @rivergold: 核心思想是使用`nn.ModuleDict`和`nn.Module.named_children()`，依次 copy 输入的`in_model`中每个 Module，直到找到了所有需要的 layer。

```python
from collections import OrderedDict
import torch
import torch.nn as nn

class InternalLayerGetter(nn.ModuleDict):
    def __init__(self, in_model, internal_layer_names:dict):
        """Get in_model internal layer output feature

        Arguments:
            in_model {nn.Module} -- Input model
            internal_layer_names {dict} -- [description]

        Raises:
            ValueError: [description]
        """
        # Check
        if not set(internal_layer_names.keys()).issubset([name for name, _ in in_model.named_children()]):
            raise ValueError('Internal layer not in model, please check again.')

        self._internal_layer_names = internal_layer_names.copy()
        tmp_internal_layer_names = internal_layer_names.copy()
        modules = OrderedDict()

        for name, module in in_model.named_children():
            modules[name] = module
            if name in tmp_internal_layer_names.keys():
                tmp_internal_layer_names.pop(name)
            if not tmp_internal_layer_names:
                break
        super().__ini__(modules)

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self._internal_layer_names.keys():
                out[self._internal_layer_names[name]] = x
        return out
```
