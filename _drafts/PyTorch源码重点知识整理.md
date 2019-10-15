# 问题

## 弱引用指针？ _from `c10/util/intrusive_ptr.h`_

弱引用指针： 没有“所有权”的指针;

只是指向 target 的指针，不会影响 target 是否被释放

强引用指针管理 target 何时释放，弱引用指针为 user 提供获取 target 的方式

**_References:_**

- [知乎: c++的弱引用指针到底是为什么目的引入的？原理是咋回事](https://www.zhihu.com/question/26851369/answer/34271911)

- [stackoverflow: When is std::weak_ptr useful?](https://stackoverflow.com/a/21877073/4636081)

---

## 虚函数？ 纯虚函数？ _from C++_

<!--  -->
<br>

---

<br>
<!--  -->

# C10

## `intrusive_ptr_target`, `intrusive_ptr`和`weak_intrusive_ptr`

声明、定义在`c10/util/intrusive_ptr.h`

- `intrusive_ptr_target`含有引用计数变量的 target 类，如果之后需要引用计数的类，都需要继承自`intrusive_ptr_target`

- `intrusive_ptr`指向`intrusive_ptr_target`的强指针类

- `weak_intrusive_ptr`指向`intrusive_ptr_target`的弱指针类

- `intrusive_ptr`和`weak_intrusive_ptr`指向的都是`intrusive_ptr_target`或者是其子类

<!--  -->
<br>

---

<br>
<!--  -->

# ATen

## `at::Tensor`

声明在`aten/src/ATen/core/Tensor.h`

- 采用 Bridge 模式：这里的 Tensor 仅提供接口，具体的函数功能实现由`protected`成员`c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;`实现

<!--  -->
<br>

---

<br>
<!--  -->

# Registry and Dispatch

## DeviceType

- `CPUType`
- `CUDAType`
- `MkldnnCPUType`
- `SparseCPUType`
- ...

声明、定义(这里的定义只是接口)在`build/aten/src/ATen`中。由`codegen.cmake`根据`aten/src/ATen/templates`在编译过程中生成的

- VariableType

声明、定义(这里的定义只是接口)在`torch/csrc/autograd/generated/`。**什么时候生成的？**。`VariableType`中的函数是带有 autograd 的功能的。

> **From Gemfield:**
> 其中 cpp 文件目前 hardcode 了 5 个：VariableType_0.cpp ～ VariableType_4.cpp，所有的函数被 Hash 到了这 5 个里面（为了加快编译速度而将一个文件拆分成了 5 个）。VariableTypeEverything.cpp 文件包含了 0 ～ 4 所有的函数，是为了给人看的（并不参与编译）。
> VariableType 系列文件定义了 VariableType 类，该类继承了 at::Type，就像在 Gemfield：PyTorch ATen 代码的动态生成 一文中介绍的那样，该类是 Type 继承体系的一个子类（类比于 CUDATypeDefault、CPUTypeDefault 等），同其它的子类最明显的区别是，VariableType 实现了 operators 自动求微分的功能。 当一个 Tensor 具备 autograd 功能时（requires_grad=True)，也就是这个 Tensor 是个 Variable，那么在进入 dispatch 前，首先要经历 Variable 到 VariableType 的分发——也就是 Variable 上的计算会首先转发到 VariableType 类上。

**_References:_**

- :thumbsup:[知乎-Gemfield: PyTorch Autograd 代码的动态生成](https://zhuanlan.zhihu.com/p/56924766)
- :thumbsup:[知乎-Gemfield: PyTorch ATen 代码的动态生成](https://zhuanlan.zhihu.com/p/55966063)

---

## Registry

### Old Way `globalATenDispatch`

- 一个函数名对应了多种针对不同 Type 版本的实现

- 注册索引表， <backend, function_impl>

- 所有函数的`namespace`都为`at`

- :triangular_flag_on_post:非`VariableType.cpp`通过调用`static auto& registerer = globalATenDispatch().registerOp ...`进行注册， `VariableType.cpp`通过调用`static auto& registerer = globalATenDispatch().registerVariableOp`进行注册

对于

- `CPUType.cpp`
- `CUDAType.cpp`
- `MkldnnCPUType.cpp`
- `SparseCPUType.cpp`
- ...

注册了不同 Type 的函数，底层实现为`at::native::<function_name>_<backend>`

对于

- `VariableType.cpp`

注册了带有 autograd 功能的 Type 函数，底层实现为`at::<function_name>`

:triangular_flag_on_post:`VariableType`中的函数之后会采用 dispatch 机制，根据 backend，调用到对应的已经注册好的`CPUType.cpp`，`CUDAType.cpp`等文件里面的函数。

### New Way `c10 dispatcher`

在 v1.2.0 之后开发

#### Register ATen ops with c10, use c10 dispatcher

- [[Issue] Plan for Migrating ATen ops to the c10 dispatcher #24132](https://github.com/pytorch/pytorch/issues/24132)
- :triangular_flag_on_post:[[Pull Request, merged] Register ATen ops with c10 #26131](https://github.com/pytorch/pytorch/pull/26131)
- [[Pull Request, not merge yet] Make schema part of RegisterOperators::Options #26114](https://github.com/pytorch/pytorch/pull/26114)

<!--  -->
<br>

---

<br>
<!--  -->

# Tensor

`Parameters(Python) -> torch.Tensor(Python) -> torch._C._TensorBase(wrapped C++) -> THPVariable(wrapped C++) ->(包含) -> torch::autograd::Variable(C++) -> at::Tensor(C++)`

**_References:_**

- [知乎-Gemfield: PyTorch 的 Tensor（上）](https://zhuanlan.zhihu.com/p/54896021)

<!--  -->
<br>

---

<br>
<!--  -->

# Autograd

**_References:_**

- [知乎-Gemfield: PyTorch 的 Tensor(中)](https://zhuanlan.zhihu.com/p/64135058)
- [PyTorch internals: Autograd](http://blog.ezyang.com/2019/05/pytorch-internals/)

<!--  -->
<br>

---

<br>
<!--  -->

# 最新理解

## RegisterOperators

E.g.

```c++
static auto registerer = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("aten::abs_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &CPUType::abs_>(TensorTypeId::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
  .op // ...
```

`c10::RegisterOperators`:

```c++
  /**
   * Call this to register an operator. See class doc comment for examples.
   */
  RegisterOperators&& op(Options&& options) && {
    checkSchemaAndRegisterOp_(std::move(options));
    return std::move(*this);
  }
```

`options`是`c10::RegisterOperators`的`static`函数，其返回一个`Options`的实例化对象

`schema`, `impl_unboxedOnlyKernel`和`aliasAnalysis`都是`Options`类的方法，这些函数的返回类型都是右值引用`Options &&`

...

注册后的结果是，所有注册的函数都会被添加到`namespace c10`下的`std::vector<OperatorRegistrar> registrars_;`中，之后由`c10::Dispatch`进行分发

`RegisterOperators`和`Options`的注册过程均采用右值引用的方式，这样做的目的是减少拷贝，且在其作用完成后，就会被释放。

<!--  -->
<br>

---

<br>
<!--  -->

# TH to ATen

## 参考

### `pow`的实现

- [Issue: Port `pow` operator from the TH code to Aten #23492](https://github.com/pytorch/pytorch/pull/23492)
- [PR: Migrate `pow` and `pow_` from the TH to Aten (CPU) #24750](https://github.com/pytorch/pytorch/issues/24750)
