# 问题

## 弱引用指针？ _from `c10/util/intrusive_ptr.h` _

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

## `intrusive_ptr_target` , `intrusive_ptr` 和 `weak_intrusive_ptr`

声明、定义在 `c10/util/intrusive_ptr.h`

- `intrusive_ptr_target` 含有引用计数变量的 target 类，如果之后需要引用计数的类，都需要继承自 `intrusive_ptr_target`

- `intrusive_ptr` 指向 `intrusive_ptr_target` 的强指针类

- `weak_intrusive_ptr` 指向 `intrusive_ptr_target` 的弱指针类

- `intrusive_ptr` 和 `weak_intrusive_ptr` 指向的都是 `intrusive_ptr_target` 或者是其子类

<!--  -->
<br>

---

<br>
<!--  -->

# ATen

## `at::Tensor`

声明在 `aten/src/ATen/core/Tensor.h`

- 采用 Bridge 模式：这里的 Tensor 仅提供接口，具体的函数功能实现由 `protected` 成员 `c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;` 实现

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

声明、定义(这里的定义只是接口)在 `build/aten/src/ATen` 中。由 `codegen.cmake` 根据 `aten/src/ATen/templates` 在编译过程中生成的

- VariableType

声明、定义(这里的定义只是接口)在 `torch/csrc/autograd/generated/` 。**什么时候生成的？**。 `VariableType` 中的函数是带有 autograd 的功能的。

> **From Gemfield:**
> 其中 cpp 文件目前 hardcode 了 5 个：VariableType_0.cpp ～ VariableType_4.cpp，所有的函数被 Hash 到了这 5 个里面（为了加快编译速度而将一个文件拆分成了 5 个）。VariableTypeEverything.cpp 文件包含了 0 ～ 4 所有的函数，是为了给人看的（并不参与编译）。
> VariableType 系列文件定义了 VariableType 类，该类继承了 at:: Type，就像在 Gemfield：PyTorch ATen 代码的动态生成 一文中介绍的那样，该类是 Type 继承体系的一个子类（类比于 CUDATypeDefault、CPUTypeDefault 等），同其它的子类最明显的区别是，VariableType 实现了 operators 自动求微分的功能。 当一个 Tensor 具备 autograd 功能时（requires_grad=True)，也就是这个 Tensor 是个 Variable，那么在进入 dispatch 前，首先要经历 Variable 到 VariableType 的分发——也就是 Variable 上的计算会首先转发到 VariableType 类上。

**_References:_**

- :thumbsup:[知乎-Gemfield: PyTorch Autograd 代码的动态生成](https://zhuanlan.zhihu.com/p/56924766)
- :thumbsup:[知乎-Gemfield: PyTorch ATen 代码的动态生成](https://zhuanlan.zhihu.com/p/55966063)

---

## Registry

### Old Way `globalATenDispatch`

- 一个函数名对应了多种针对不同 Type 版本的实现

- 注册索引表， <backend, function_impl>

- 所有函数的 `namespace` 都为 `at`

- :triangular_flag_on_post: 非 `VariableType.cpp` 通过调用 `static auto& registerer = globalATenDispatch().registerOp ...` 进行注册， `VariableType.cpp` 通过调用 `static auto& registerer = globalATenDispatch().registerVariableOp` 进行注册

对于

- `CPUType.cpp`
- `CUDAType.cpp`
- `MkldnnCPUType.cpp`
- `SparseCPUType.cpp`
- ...

注册了不同 Type 的函数，底层实现为 `at::native::<function_name>_<backend>`

对于

- `VariableType.cpp`

注册了带有 autograd 功能的 Type 函数，底层实现为 `at::<function_name>`

:triangular_flag_on_post: `VariableType` 中的函数之后会采用 dispatch 机制，根据 backend，调用到对应的已经注册好的 `CPUType.cpp` ， `CUDAType.cpp` 等文件里面的函数。

### New Way `c10 dispatcher`

在 v1.2.0 之后开发

#### Register ATen ops with c10, use c10 dispatcher

- [[Issue] Plan for Migrating ATen ops to the c10 dispatcher #24132](https://github.com/pytorch/pytorch/issues/24132)
- :triangular_flag_on_post:[[Pull Request, merged] Register ATen ops with c10 #26131](https://github.com/pytorch/pytorch/pull/26131)
- [[Pull Request, not merge yet] Make schema part of RegisterOperators:: Options #26114](https://github.com/pytorch/pytorch/pull/26114)

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
static auto registerer = torch:: RegisterOperators()
  .op(torch:: RegisterOperators::options()

    .schema("aten::abs_(Tensor(a!) self) -> Tensor(a!)")
    .impl_unboxedOnlyKernel<Tensor & (Tensor &), &CPUType::abs_>(TensorTypeId::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))

  .op // ...

```

`c10::RegisterOperators` :

```c++
  /**
   * Call this to register an operator. See class doc comment for examples.
   */
  RegisterOperators&& op(Options&& options) && {
    checkSchemaAndRegisterOp_(std::move(options));
    return std::move(*this);
  }
```

`options` 是 `c10::RegisterOperators` 的 `static` 函数，其返回一个 `Options` 的实例化对象

`schema` , `impl_unboxedOnlyKernel` 和 `aliasAnalysis` 都是 `Options` 类的方法，这些函数的返回类型都是右值引用 `Options &&`

...

注册后的结果是，所有注册的函数都会被添加到 `namespace c10` 下的 `std::vector<OperatorRegistrar> registrars_;` 中，之后由 `c10::Dispatch` 进行分发

`RegisterOperators` 和 `Options` 的注册过程均采用右值引用的方式，这样做的目的是减少拷贝，且在其作用完成后，就会被释放。

<!--  -->
<br>

---

<br>
<!--  -->

# TH to ATen

## 参考

### `pow` 的实现

- [Issue: Port `pow` operator from the TH code to Aten #23492](https://github.com/pytorch/pytorch/pull/23492)
- [PR: Migrate `pow` and `pow_` from the TH to Aten (CPU) #24750](https://github.com/pytorch/pytorch/issues/24750)

<!--  -->
<br>

---

<br>
<!--  -->

# Build PyTorch with clang

**_References:_**

- [PyTorch Forum: Current way to compile from source](https://discuss.pytorch.org/t/current-way-to-compile-from-source/19635)

<!--  -->
<br>

---

<br>
<!--  -->

# AQ

## `scalar_t`

e.g.in `aten/src/ATen/native/cpu/PowKernel.cpp`

TODO:

<!--  -->
<br>

---

<br>
<!--  -->

# Build from PyTorch Source

**_References:_**

- [PyTorch Home](https://pytorch.org/get-started/locally/)

## 编译时间

- CPU: 8 core

---

## Third Party

### MKL-DNN

**_References:_**

- [Github intel/mkl-dnn: who can explain the association among MKL, MKLML, MKLDNN #102](https://github.com/intel/mkl-dnn/issues/102)
- [ApacheMXNet 博客: 用 Intel MKL-DNN 加速 CPU 上的深度学习](https://zh.mxnet.io/blog/mkldnn)

---

## 体会

- ninja 相比于 make 会提高编译速度
- ccache 在二次（多次编译）时会明显节省编译时间，其原因在于 ccache 在编译初期存储了编译过程中的信息

---

## Problem & Solution

### [Git submodule] Build third_party occur error

E.g.

```shell
cd /root/rivergold-project/pytorch/build/confu-deps/NNPACK && PYTHONPATH=/root/rivergold-project/pytorch/third_party/python-six:/root/rivergold-project/pytorch/third_party/python-peachpy /root/software/anaconda/bin/python -m peachpy.x86_64 -mabi=sysv -g4 -mimage-format=elf -I/root/rivergold-project/pytorch/third_party/NNPACK/src -I/root/rivergold-project/pytorch/third_party/NNPACK/src/x86_64-fma -I/root/rivergold-project/pytorch/third_party/FP16/include -o /root/rivergold-project/pytorch/build/confu-deps/NNPACK/src/x86_64-fma/2d-fourier-8x8.py.o /root/rivergold-project/pytorch/third_party/NNPACK/src/x86_64-fma/2d-fourier-8x8.py
Traceback (most recent call last):
  File "/root/software/anaconda/lib/python3.7/runpy.py", line 183, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/root/software/anaconda/lib/python3.7/runpy.py", line 142, in _get_module_details
    return _get_module_details(pkg_main_name, error)
  File "/root/software/anaconda/lib/python3.7/runpy.py", line 109, in _get_module_details
    __import__(pkg_name)
  File "/root/rivergold-project/pytorch/third_party/python-peachpy/peachpy/x86_64/__init__.py", line 25, in <module>
    from peachpy.x86_64.function import Function, LocalVariable
  File "/root/rivergold-project/pytorch/third_party/python-peachpy/peachpy/x86_64/function.py", line 16, in <module>
    import peachpy.x86_64.avx
ModuleNotFoundError: No module named 'peachpy.x86_64.avx'
```

**Solution:**

TODO:

### [CMake Configuring] Could not find OpenMP

```shell
-- Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
-- Could NOT find OpenMP (missing: OpenMP_C_FOUND OpenMP_CXX_FOUND)
CMake Error at third_party/ideep/mkl-dnn/cmake/OpenMP.cmake:115 (message):
  OpenMP library could not be found.  Proceeding might lead to highly
  sub-optimal performance.
Call Stack (most recent call first):
  third_party/ideep/mkl-dnn/CMakeLists.txt:76 (include)

-- Configuring incomplete, errors occurred!
```

_*Solution:*_

TODO:

### [Use PyTorch] cmake cannot find `public/xxx.cmake`

```cmake
CMake Error at /home/ubtuntu/pytorch/build/Caffe2Config.cmake:14 (include):
  include could not find load file:

    /home/ubtuntu/pytorch/build/public/utils.cmake
```

Same error from website:

- [Github facebookresearch/Detectron: make ops fails with pytorch compiled from source #715](https://github.com/facebookresearch/Detectron/issues/715)
- [Github pytorch/pytorch: [Caffe2] Caffe2Config.cmake #15009](https://github.com/pytorch/pytorch/issues/15009)

**Solution:**

TODO:

<!--  -->
<br>

---

<br>
<!--  -->

# PyTorch

## Config PyTorch in CMake

cmake need to find `TorchConfig.cmake` to config PyTorch, `TorchConfig.cmake` often in path `<pytorch_root_dir>/torch/share/cmake/Torch`

CMakeLists.txt

```cmake
# PyTorch
set(CMAKE_PREFIX_PATH /root/rivergold-project/pytorch/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
message("Torch Found: ${TORCH_FOUND}")
message("Torch include_dir: ${TORCH_INCLUDE_DIRS}")
message("Torch libs: ${TORCH_LIBRARIES}")
include_directories(${TORCH_INCLUDE_DIRS})

list(APPEND SRC test_pytorch.cc)

add_executable(test_pytorch ${SRC})
target_link_libraries(test_pytorch ${TORCH_LIBRARIES})
```

**_References:_**

- [PyTorch doc: PYTORCH C++ API](https://pytorch.org/cppdocs/)
- [Github pytorch/pytorch [Caffe2] Caffe2Config.cmake #15009](https://github.com/pytorch/pytorch/issues/15009)

<!--  -->
<br>

---

<br>
<!--  -->

# `native_functions.yaml`参数理解整理

基础格式

```yaml
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  variants: function, method
  dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```

## References

- :thumbsup::triangular_flag_on_post:[PyTorch Forum: How to create a new torch.function() method?](https://discuss.pytorch.org/t/how-to-create-a-new-torch-function-method/21899)

- [Github pytorch/pytorch: More structured alternative to native_functions.yaml #12417](https://github.com/pytorch/pytorch/issues/12417)

---

## `func`

参数类型符号(Argument types)理解:

- `Tensor`: `const Tensor&`
- `Tensor(a)` : `const Tensor&`, 接受、返回操作同一段 data 的变量
  - **理解:** 针对于 PyTorch 中 op 返回的是 Tensor view 的情况。请在`native_functions.yaml`中搜索`Tensor(a)`, 可以看到基本上都是 view 的操作
- `Tensor(a!)`: `Tensor&`, 参数的 member 数据会发生变化

### `Tensor(a)`

在`native_functions.yaml`中使用的不多，主要针对于 view 的操作，内部多使用 `std::move`并配合 Tensor 的移动构造函数

E.g.

```yaml
- func: alias(Tensor(a) self) -> Tensor(a)
  use_c10_dispatcher: unboxed_only
  variants: method, function
  supports_named_tensor: True
```

Function `Tensor alias(const Tensor& self) { ... }` in `aten/src/ATen/native/TensorShape.cpp`

注意其在最后返回的是`self_ = Tensor(std::move(impl));`

### Problem & Solution

#### [Build Error] libtorch.so: undefined reference to `at::native::rivergold_test(at::Tensor const&)`

```shell
FAILED: bin/conv_to_nnpack_transform_test
: && /root/software/ccache/install/bin/c++  -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Wno-stringop-overflow -DHAVE_AVX_CPU_DEFINITION -DHAVE_AVX2_CPU_DEFINITION -O2  -rdynamic caffe2/CMakeFiles/conv_to_nnpack_transform_test.dir/transforms/conv_to_nnpack_transform_test.cc.o  -o bin/conv_to_nnpack_transform_test  -Wl,-rpath,/root/rivergold-project/pytorch/build/lib: lib/libgtest_main.a -Wl,--no-as-needed,/root/rivergold-project/pytorch/build/lib/libtorch.so -Wl,--as-needed lib/libprotobuf.a lib/libc10.so lib/libmkldnn.a lib/libgtest.a -lpthread && :
/root/rivergold-project/pytorch/build/lib/libtorch.so: undefined reference to `at::native::rivergold_test(at::Tensor const&)'
collect2: error: ld returned 1 exit status
```

本质原因是: `native_functions.yaml`中说明的 Argument types 和 `.cpp`写的不一致，或者是写法不符合要求。查看方法：`build/aten/src/ATen/Declarations.yaml`中的函数说明，看是否是你想要的

错误举例

```yaml
- func: rivergold_test(Tensor(a!) self) -> Tensor(a!)
  dispatch:
    CPU: rivergold_test
```

```shell
>>>Debug: rivergold_test
>>>Debug return_arguments: [{'type': 'Tensor', 'name': 'result', 'annotation': 'a!', 'output': True}]
>>>Debug arguments: [{'type': 'Tensor', 'name': 'self', 'is_nullable': False, 'annotation': 'a!'}]
{'mode': 'native', 'schema_string': 'aten::rivergold_test(Tensor(a!) self) -> Tensor(a!)', 'name': 'rivergold_test', 'operator_name': 'rivergold_test', 'overload_name': '', 'inplace': False, 'return': [{'type': 'Tensor', 'name': 'result', 'annotation': 'a!', 'output': True}], 'variants': ['function'], 'requires_tensor': False, 'matches_jit_signature': True, 'cpu_half': False, 'cpu_bfloat16': False, 'cpu_bool': False, 'cuda_bool': False, 'deprecated': False, 'device_guard': True, 'supports_named_tensor': False, 'use_c10_dispatcher': 'no', 'category_override': '', 'arguments': [{'type': 'Tensor', 'name': 'self', 'is_nullable': False, 'annotation': 'a!'}], 'type_method_definition_dispatch': {'CPU': 'rivergold_test'}, 'python_module': ''}
```

```yaml
# build/aten/src/ATen/Declarations.yaml
- name: rivergold_test
  operator_name: rivergold_test
  overload_name: ""
  use_c10_dispatcher: "no"
  category_override: ""
  matches_jit_signature: true
  schema_string: aten::rivergold_test(Tensor(a!) self) -> Tensor(a!)
  method_prefix_derived: ""
  arguments:
    - annotation: a!
      dynamic_type: Tensor
      is_nullable: false
      name: self
      type: const Tensor &
  method_of:
    - Type
    - namespace
  mode: native
  python_module: ""
  returns:
    - dynamic_type: Tensor
      name: result
      type: Tensor
  inplace: false
  is_factory_method: null
  abstract: true
  requires_tensor: false
  device_guard: true
  with_gil: false
  deprecated: false
```

---

## `variants`

- By default, ATen generates only the function variant for a native function.
