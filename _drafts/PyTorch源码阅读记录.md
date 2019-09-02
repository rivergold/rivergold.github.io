# `setup.py`

- `lib_path`里面有什么
- `torch._C`都有什么

`build_caffe`会执行根目录（PyTorch）下的 cmakelist

```shell
cmake -DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/torch -DCMAKE_PREFIX_PATH=/home/rivergold/software/anaconda/envs/learn-pytorch/lib/python3.7/site-packages -DPYTHON_EXECUTABLE=/home/rivergold/software/anaconda/envs/learn-pytorch/bin/python -DPYTHON_INCLUDE_DIR=/home/rivergold/software/anaconda/envs/learn-pytorch/include/python3.7m -DPYTHON_LIBRARY=/home/rivergold/software/anaconda/envs/learn-pytorch/lib/libpython3.7m.so.1.0 -DTORCH_BUILD_VERSION=1.2.0a0+8554416 -DUSE_CUDA=False -DUSE_DISTRIBUTED=True -DUSE_NUMPY=True -DUSE_SYSTEM_EIGEN_INSTALL=OFF /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch
```

这一步会调用基本上所有的 cmakelist， 包括 thirdparty

根目录下的`CMakeLists.txt`会编译 `c10` 和 `caffe2`

```shell
# ---[ Main build
add_subdirectory(c10)
add_subdirectory(caffe2)
```

`caffe2/CMakeLists.txt`的主要编译工作：

基本顺序是：`caffe2/CMakeLists.txt`调用`cmake/Codegen.cmake`生成 target`ATEN_CPU_FILES_GEN_LIB`和`ATEN_CUDA_FILES_GEN_LIB` -> `caffe2/CMakeLists.txt`调用`aten/CMakeLists.txt` -> `aten/ATen/CMakeLists`会 link`ATEN_CPU_FILES_GEN_LIB`和`ATEN_CUDA_FILES_GEN_LIB`

- `INTERN_BUILD_ATEN_OPS`默认是 True
- `cmake/Codegen.cmake`根据`aten/src/ATen/Declarations.cwrap`生成的`aten/src/ATen/Declarations.yaml`很重要，跟`ATen`相关的编译基本都用到了 yaml 这个文件

1. `caff2/CMakeLists.txt`会先调用`cmake/Codegen.cmake`，做一些准备工作。暂时还不了解具体的工作。
   这部分的工作是根据 template 生成`aten/ATen`的代码并编译

2. 在`caffe2/CMakeLists.txt`会在编译`aten`时，会调用 python 去执行`caffe2/contrib/aten/gen_op.py`(传递了`aten`的路径)。这一步的`gen_op.py`使用了由`cmake/Codegen.cmake`调用`aten/src/ATen/gen.py`生成的`Declarations.yaml`。`aten/src/ATen/gen.py`在生成`Declarations.yaml`使用了`aten/src/ATen/native/native_functions.yaml`

   **\*References:**

   - [PyTorch Forum: How are python bindings created?](https://discuss.pytorch.org/t/how-are-python-bindings-created/46453/2?u=rivergold)

同时，在`caffe2`下的`CMakeLists.txt`会调用 python 去执行`tools/setup_helpers/generate_code.py`，生成`autograd`所需要的代码（目测生成的是 C++代码，和 python bind 没关系的代码）

之后调用`add_library(torch ${Caffe2_CPU_SRCS})`编译出`torch`(C++的，和 Python binding 没有关系)，这一步会根据情况编译 GPU 和 CPU 版本。

最后再调用`add_subdirectory(../torch torch)`来编译 python bind 的 torch，库名叫`torch_python`。`torch/CMakeLists.txt`仅编译 Torch Python binding。

`torch/csrc`为 C++代码，一部分为纯 C++代码，一部分是 C++ 和 Python 的混合编程；其目的是为了实现 python bind to C++，`torch`除`csrc`以外的文件都和 C++无关。

## `caffe2/CMakeLists.txt`

```shell
add_custom_command(
OUTPUT
${TORCH_GENERATED_CODE}
COMMAND
"${PYTHON_EXECUTABLE}" tools/setup_helpers/generate_code.py
  --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
  --nn-path "aten/src"
......
```

这里所用到的`${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml`是由`cmake/Codegen.cmake`根据`aten/ATen/Declarations.cwarp`生成出来的。`Codegen.cmake`会调用`aten/src/ATen/gen.py`生成`Declarations.yaml`。这里需要参考[知乎专栏-Gemfield PyTorch ATen 代码的动态生成](https://zhuanlan.zhihu.com/p/55966063)

```shell
cd cmake/
python ../aten/src/ATen/gen.py --source-path ../aten/src/ATen --install_dir /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen ../aten/src/ATen/Declarations.cwrap ../aten/src/THNN/generic/THNN.h ../aten/src/THCUNN/generic/THCUNN.h ../aten/src/ATen/nn.yaml ../aten/src/ATen/native/native_functions.yaml  --output-dependencies /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen/generated_cpp.txt
```

## `cmake/Codegen.cmake`

```shell
  FOREACH(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
    FOREACH(IMPL ${cpu_kernel_cpp_in})
      string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/" "" NAME ${IMPL})
      LIST(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
      SET(NEW_IMPL ${CMAKE_BINARY_DIR}/aten/src/ATen/${NAME}.${CPU_CAPABILITY}.cpp)
      CONFIGURE_FILE(${IMPL} ${NEW_IMPL} COPYONLY)
      SET(cpu_kernel_cpp ${NEW_IMPL} ${cpu_kernel_cpp}) # Create list of copies
      LIST(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
      IF(MSVC)
        SET(MACRO_FLAG "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
      ELSE(MSVC)
        SET(MACRO_FLAG "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
      ENDIF(MSVC)
      SET_SOURCE_FILES_PROPERTIES(${NEW_IMPL} PROPERTIES COMPILE_FLAGS "${FLAGS} ${MACRO_FLAG}")
    ENDFOREACH()
  ENDFOREACH()
  list(APPEND ATen_CPU_SRCS ${cpu_kernel_cpp})
```

这里赋值的`ATen_CPU_SRCS`会在多个地方使用

`CONFIGURE_FILE(input_file, output_file)`

这里会生成`build/aten/src/ATen/native/cpu/<name1>.cpp.<name2>.cpp`文件：

- `name1`: 对应与`aten/src/ATen/native/cpu/*.cpp`的文件名
- `name2`: `DEFAULT`, `AVX2`等

```shell
├── Activation.cpp
├── avx_mathfun.h
├── BinaryOpsKernel.cpp
├── CopyKernel.cpp
├── CrossKernel.cpp
├── DistanceOpsKernel.cpp
├── FillKernel.cpp
├── GridSamplerKernel.cpp
├── GridSamplerKernel.h
├── IndexKernel.cpp
├── Intrinsics.h
├── IsContiguous.h
├── layer_norm_kernel.cpp
├── layer_norm_kernel.h
├── LerpKernel.cpp
├── Loops.h
├── README.md
├── Reduce.h
├── ReduceOpsKernel.cpp
├── SoftMaxKernel.cpp
├── SoftmaxKernel.h
├── SortingKernel.cpp
├── TensorCompareKernel.cpp
├── TensorCompareKernel.h
└── UnaryOpsKernel.cpp
```

```shell
  SET(GEN_COMMAND
      "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/gen.py
      --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
      --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
      ${GEN_ROCM_FLAG}
      ${cwrap_files}
  )
```

生成`build/aten/src/ATen`

```shell
├── core_tmp
│   ├── Tensor.h
│   └── TensorMethods.h
├── CPUType.cpp
├── CPUType.h
├── CUDAType.cpp
├── CUDAType.h
├── Declarations.yaml
├── ExtensionBackendRegistration.h
├── Functions.h
├── generated_cpp.txt
├── generated_cpp.txt-core
├── generated_cpp.txt-cuda
├── LegacyTHFunctionsCPU.cpp
├── LegacyTHFunctionsCPU.h
├── LegacyTHFunctionsCUDA.cpp
├── LegacyTHFunctionsCUDA.h
├── MkldnnCPUType.cpp
├── MkldnnCPUType.h
├── MSNPUType.cpp
├── MSNPUType.h
├── NativeFunctions.h
├── QuantizedCPUType.cpp
├── QuantizedCPUType.h
├── RegistrationDeclarations.h
├── SparseCPUType.cpp
├── SparseCPUType.h
├── SparseCUDAType.cpp
├── SparseCUDAType.h
├── TypeDefault.cpp
├── TypeDefault.h
├── XLAType.cpp
└── XLAType.h
```

# 问题

- [ ] dispatch mechanism
- [x] add_custom_target
- [ ] 侵入式指针
- [ ] weak references
- [ ] `globalATenDispatch`

# 零散的理解

PyTorch 的对于 Tensor operations 的代码主要为`c10`, `ATen`和原始`torch`的`TH*`的代码

**_References:_**

- [PyTorch Forum: The pytorch blog “A Tour of PyTorch Internals” is out-of-date. How to know more about the pytorch internal](https://discuss.pytorch.org/t/the-pytorch-blog-a-tour-of-pytorch-internals-is-out-of-date-how-to-know-more-about-the-pytorch-internal/19677/4?u=rivergold)

- [知乎专栏 - Gemfield: PyTorch ATen 代码的动态生成](https://zhuanlan.zhihu.com/p/55966063)

---

目录结构

**_References:_**

- [Oldpan 博客: Pytorch 源码编译简明指南](https://oldpan.me/archives/pytorch-build-simple-instruction)

---

`torch._C`是 Python `import torch`的最重要的一部分，该 module 的底层 C++代码是在`torch/src/Module.cpp`中定义的

```c++
#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", methods.data()));
#else
  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     nullptr,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
```

**_References:_**

- :thumbsup:[PyTorch Forum: Where does `torch._C` come from?](https://discuss.pytorch.org/t/where-does-torch-c-come-from/2015/3?u=rivergold)

---

`caffe2/CMakeLists.txt`中编译出的`libtorch.so`是纯 C++的 torch 库，之后会调用`add_subdirectory(../torch torch)`运行`torch/CMakeLists.txt`来编译 torch 的 Python binding 的 C++库，名叫`libtorch_python`。`libtorch.so`是`libtorch_python`的依赖库

C10 -> aten -> Caffe2 -> torch -> torch_python

---

C10，ATen 的关系

**_References:_**

- [知乎专栏 - Gemfield: PyTorch 的库文件](https://zhuanlan.zhihu.com/p/57437931)

---

PyTorch Tensor

Python `class Tensor(torch._C._TensorBase)`继承`_TensorBase`, `_TensorBase` binding C++的`THPVariable`，`THPVariable`封装的`torch::autograd::Variable`, `torch::autograd::Variable`继承的`at::Tensor`，`at::Tensor`定义在`aten/ATen/Core/Tensor.h`

我的理解：`at::Tensor`是个接口类，基于 PIml 思想，其具体实现由`TensorImpl`实现。`TensorImpl`定义在`c10/core/TensorImpl.h`中。`TensorImpl`继承`c10::intrusive_ptr_target`

`c10::intrusive_ptr_target`是一种侵入式指针，即其引用计数是放在其指向的 object 的 member 的

**_References:_**

- [知乎专栏 - Gemfield: PyTorch 的 Tensor（上）](https://zhuanlan.zhihu.com/p/54896021)
- [知乎专栏 - Gemfield: PyTorch 的 Tensor（中）](https://zhuanlan.zhihu.com/p/64135058)
- [知乎专栏 - Gemfield: PyTorch 的 Tensor（下）](https://zhuanlan.zhihu.com/p/69530008)

<!--  -->
<br>

---

<br>
<!--  -->

# 一些 C++的基础库

## gflags

Google 开源的用于命令行参数处理的库

## glog

Google 开源的用于实现应用级别的 logging 的库

## NUMA

Non-uniform memory access

一种为多处理器的计算机设计的内存架构

**_References:_**

- [Wiki: 非均匀访存模型](https://zh.wikipedia.org/wiki/%E9%9D%9E%E5%9D%87%E5%8C%80%E8%AE%BF%E5%AD%98%E6%A8%A1%E5%9E%8B)

# 边编译边了解

## `python setup.py build`

- [ ] upload `build.log`

## `aten/src/ATen/gen.py`

```shell
python ../aten/src/ATen/gen.py --source-path ../aten/src/ATen --install_dir /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen ../aten/src/ATen/Declarations.cwrap ../aten/src/THNN/generic/THNN.h ../aten/src/THCUNN/generic/THCUNN.h ../aten/src/ATen/nn.yaml ../aten/src/ATen/native/native_functions.yaml  --output-dependencies /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen/generated_cpp.txt
```

## `gen_op.py`

```shell
python /home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/caffe2/contrib/aten/gen_op.py --aten_root=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/aten --template_dir=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/caffe2/contrib/aten --yaml_dir=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/aten/src/ATen --install_dir=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/tmp_build/contrib/aten --aten_root=/home/rivergold/Documents/RiverGold/Learn-From-Src/pytorch/aten
```

# 重点知识总结

## 主要目录

- c10
- aten
- caffe2
- torch

目前 PyTorch 的核心算子都在 aten 的 ATen 中

torch 中定义大部分的 autograd，以及 Python 如何调用底层的 aten

## Dispatch

- Device
- Type

# `aten/src/ATen/native`

`native_functions.yaml`注册了 ATen 中的函数，
`dispatch`定义了如何分发到具体的函数

**_References:_**

- [pytorch/aten/src/ATen/native/README.md](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md)

## `aten/src/ATen/gen.py`

会生成

```shell
ATen/NativeFunctions.h
ATen/LegacyTHFunctionsCPU.h

```

<!--  -->
<br>

---

<br>
<!--  -->

# Python bind C++

## `PyMethodDef`

`torch/csrc/Module.cpp`

```c++
static PyMethodDef TorchMethods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,   METH_O,       nullptr},
  {"_autograd_init",  (PyCFunction)THPAutograd_initExtension, METH_NOARGS,  nullptr},
  {"_add_docstr",     (PyCFunction)THPModule_addDocStr,       METH_VARARGS, nullptr},
  {"_init_names",     (PyCFunction)THPModule_initNames,       METH_O,       nullptr},
  {"_has_distributed",(PyCFunction)THPModule_hasDistributed,  METH_NOARGS,  nullptr},
...
```

`"_initExtension"`是在 Python 的函数名，`THPModule_initExtension`是封装了对应 C++函数的`PyObject`

```c++
// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *_unused, PyObject *shm_manager_path)
{
  HANDLE_TH_ERRORS
  if (!THPUtils_checkString(shm_manager_path)) {
    THPUtils_setError("initialization error - expected bytes/string object as shm_manager_path!");
    return nullptr;
  }
...
```

**_References:_**

- [博客园: 使用 C 语言扩展 Python(一)](https://www.cnblogs.com/phinecos/archive/2010/05/17/1737033.html)

---

## `PyModuleDef`

```c++
static struct PyModuleDef torchmodule = {
    PyModuleDef_HEAD_INIT,
    "torch._C",
    nullptr,
    -1,
    methods.data()
};
ASSERT_TRUE(module = PyModule_Create(&torchmodule));
```

`torch._C`为模块名，其方法有`methods`

**_References:_**

- [Blog: 为 PYTHON 写 C 的扩展](https://jmpews.github.io/2016/12/10/python/%E4%B8%BApython%E5%86%99C%E7%9A%84%E6%89%A9%E5%B1%95/)

---

## Python 端`torch`添加 wrapped C++

`torch/csrc/Module.cpp`

```c++
THPSize_init(module);
```

`torch/csrc/Size.cpp`

```c++
PyTypeObject THPSizeType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.Size",                          /* tp_name */
  sizeof(THPSize),                       /* tp_basicsize */
  0,                                     /* tp_itemsize */
  nullptr,                                     /* tp_dealloc */
  nullptr,                                     /* tp_print */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
// ...
}

void THPSize_init(PyObject *module)
{
  if (PyType_Ready(&THPSizeType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPSizeType);
  if (PyModule_AddObject(module, "Size", (PyObject*)&THPSizeType) < 0) {
    throw python_error();
  }
}
```

**总结：** 原始的 Python bind C++的过程是：

1. 包装对应的 C++函数成为`PyObject`
2. 调用`PyMethodDef`注册`PyObject`为 Python 的一个函数
3. 调用`PyModuleDef`注册多个`PyMethodDef`成为`Module`
4. 调用`PyModule_Create`生成`Module`
5. 可以调用`PyModule_ADDObject`向第 4 步生成的`Module`中继续添加`PyObject`

Python 体系中的 Tensor 和 Function 会继承了 warp C++之后的`torch._C._TensorBase`和`torch._C._FunctionBase`

---

## c10 注册机制

`c10/util/Registry.h`

`#define C10_DECLARE_REGISTRY` -> `#define C10_DECLARE_TYPED_REGISTRY`: 实例化`c10::Registry`

`C10_REGISTER_CREATOR` -> `C10_REGISTER_TYPED_CREATOR`: 实例化`static`的`c10::Registerer`

```c++
#define C10_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_DEFINE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_REGISTER_CREATOR(RegistryName, key, ...) \
  C10_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

// ......

#define C10_DECLARE_TYPED_REGISTRY(                                        \
    RegistryName, SrcType, ObjectType, PtrType, ...)                       \
  C10_IMPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* \
  RegistryName();                                                          \
  typedef ::c10::Registerer<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>   \
      Registerer##RegistryName

#define C10_DEFINE_TYPED_REGISTRY(                                         \
    RegistryName, SrcType, ObjectType, PtrType, ...)                       \
  C10_EXPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>* \
  RegistryName() {                                                         \
    static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*   \
        registry = new ::c10::                                             \
            Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>();       \
    return registry;                                                       \
  }

// Note(Yangqing): The __VA_ARGS__ below allows one to specify a templated
// creator with comma in its templated arguments.
#define C10_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                  \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(), ##__VA_ARGS__);

#define C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                           \
    RegistryName, key, priority, ...)                                       \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, priority, RegistryName(), ##__VA_ARGS__);
```

---

# Ops Dispath

`aten/src/ATen/native/DispatchStub.h`

> 在 PyTorch 的运行中，tensor 之间的加法会调用到 add_stub，并被分发到上述定义的 add_kernel 函数上

```C++
// Implements instruction set specific function dispatch.
//
// Kernels that may make use of specialized instruction sets (e.g. AVX) are
// compiled multiple times with different compiler flags (e.g. -mavx). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// cpuinfo.

// Example:
//
// In native/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   DECLARE_DISPATCH(fn_type, stub);
//
// In native/MyKernel.cpp
//   DEFINE_DISPATCH(stub);
//
// In native/cpu/MyKernel.cpp:
//   namespace {
//     // use anonymous namespace so that different cpu versions won't conflict
//     void kernel(const Tensor& x) { ... }
//   }
//   REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(kCPU, tensor);
```

# 尝试理解

## Python 端的`torch.add`调用到后端的过程

1.  `import torch`会`import torch._C`
2.  `torch._C`中有名叫`_TensorBase`的 PyObject
    `torch/csrc/autograd/python_variable.cpp`:

    ```c++
    PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
    ```

3.  `THPVariableType`这个 PyObject 中的`tp_methods`注册了`add`方法

    `torch/csrc/autograd/python_variable.cpp`

    ```c++
    static std::vector<PyMethodDef> methods;
    THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
    THPUtils_addPyMethodDefs(methods, extra_methods);
    THPVariableType.tp_methods = methods.data();
    ```

    `torch/csrc/autograd/python_variable_methods.cpp`

    ```c++
    PyMethodDef variable_methods[] = {
    // ...
     {"abs", (PyCFunction)THPVariable_abs, METH_NOARGS, NULL},
    // ...
    ```

4.  `THPVariable_abs`是对`dispatch_add_`的封装

    ```c++
    static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
    {
    HANDLE_TH_ERRORS
    static PythonArgParser parser({
        "add(Scalar alpha, Tensor other)|deprecated",
        "add(Tensor other, *, Scalar alpha=1)",
    }, /*traceable=*/true);
    auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
    ParsedArgs<3> parsed_args;
    auto r = parser.parse(args, kwargs, parsed_args);

    if (r.idx == 0) {
        return wrap(dispatch_add(self, r.scalar(0), r.tensor(1)));
    } else if (r.idx == 1) {
        return wrap(dispatch_add(self, r.tensor(0), r.scalar(1)));
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
    }
    ```

5.  `dispatch_add_`

    `torch/csrc/autograd/generated/python_variable_methods.cpp`

    ```c++
    inline Tensor dispatch_add_(Tensor & self, const Tensor & other, Scalar alpha) {

    AutoNoGIL no_gil;
    return self.add_(other, alpha);
    }
    ```

6.  `dispatch_add_`会调用`Tensor.add`

    `torch/csrc/autograd/generated/python_variable_methods_dispatch.h`

7.  `Tensor.add`函数定义在`aten/src/ATen/core/TensorMethods.h`(该文件在 CMakeLists 编译时，还会被拷贝到`build/aten/src/ATen/core_tmp/TensorMethods.h`)

    ```c++
    inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) {
        static auto table = globalATenDispatch().getOpTable("aten::add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)");
        return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
    }
    ```

8.  `globalATenDispath()`定义在`aten/src/ATen/core/ATenDispatch.cpp`，采用函数内的`static`（局部 static 变量）实现的单例模式

    ```c++
    #include <ATen/core/ATenDispatch.h>

    namespace at {

    ATenDispatch & globalATenDispatch() {
    static ATenDispatch singleton;
    return singleton;
    }

    } // namespace at
    ```

9.  `ATenDispatch`定义在`aten/src/ATen/core/ATenDispatch.h`中，该 dispatch 将会逐步被 c10 dispatcher 取代，从`v1.2.0`之后，开发者`smessmer`和`li-roy`已经开始着手进行该工作了

    Github issues and pull request:

    - [Plan for Migrating ATen ops to the c10 dispatcher #24132](https://github.com/pytorch/pytorch/issues/24132)

    ```c++
    // This dispatch class serves as a replacement for our previous dispatch
    // mechanism, in which all functions were members of a Type class. A derived
    // class existed for each backend (and Variable), and the vtable was used to
    // dispatch to the correct implementation. This class is to be replaced by
    // the c10 dispatcher when it supports all argument and return types.
    // This implementation opts to store implementations in a table of void*.

    // ATenOpTable stores the implementations for each backend, in addition to
    // an implementation for variables.
    class CAFFE2_API ATenOpTable {
        // ...
    ```

    问题：

    - [ ] 为什么要做这项工作？会提速吗？针对于移动端？
    - [ ] c10 中的 two types of operators - boxed ones and unboxed ones 指的是什么
          **_References:_**
      - [Medium: Deep Learning meets PyTorch (part-2)](https://medium.com/@duyanhnguyen_38925/deep-learning-meets-pytorch-part-2-1524a4345aa9)


    对于PyTorch的dispatch机制，请到[这里]()了解

10. Variable 的 Ops 在哪注册的？

**这里所说的是 CPU 的函数的注册, GPU 下的同理**

`torch/csrc/autograd/generated/VariableTypeEverything.cpp`

```c++
static auto& registerer = globalATenDispatch()
  .registerVariableOp<Tensor (const Tensor &, Scalar)>("aten::__and__(Tensor self, Scalar other) -> Tensor", &VariableType::__and__)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::__and__(Tensor self, Tensor other) -> Tensor", &VariableType::__and__)
// ......
```

这里调用`globalATenDispatch`依次注册`VariableType`中定义的 Ops 函数

struct `VariableType`定义在`torch/csrc/autograd/generated/VariableType.h`中

11. Variable 的 Ops 在哪实现的？

`torch/csrc/autograd/generated/VariableType_0`, `torch/csrc/autograd/generated/VariableType_1`, `torch/csrc/autograd/generated/VariableType_2`, `torch/csrc/autograd/generated/VariableType_3`, `torch/csrc/autograd/generated/VariableType_4`这 5 个文件分别按部分实现了 Variable 的所有 Ops。这 5 个 cpp 文件是以`tools/autograd/templates/VariableType.cpp`为模板 codegen 生成的。

12. Variable 的 Ops 是如何实现的？

整体流程参考[PyTorch internals: Anatomy of a kernel](http://blog.ezyang.com/2019/05/pytorch-internals/)

**理解：** 除去一些前处理和后处理，其核心代码还是使用的`at`的`Tensor`操作

### 对 Python 调用到 C++底层的举例

Python 端常用的代码

```python
import torch
torch.nn.functional.conv1d()
```

该函数定义在`torch/nn/functional.py`中：

```python
conv1d = _add_docstr(torch.conv1d, r"""
```

Python 的`torch`module 在`torch/__init__.py`中运行了`from torch._C import *`

**如何调用到底层的？或者说是如何 dispatch 的？**

首先给出结论，Python 端`torch.nn.functional.conv1d()`调用的是`aten/src/ATen/native/Convolution.cpp`里的:

```c++
at::Tensor conv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  return at::convolution(input, weight, bias, stride, padding, dilation,
                         false, {{0, 0}}, groups);
}
```

`at::convolution`在`torch/include/ATen/Functions.h`(该文件是由`aten/src/ATen/gen.py`生成的)，该文件声明、定义了 ATen 的 Functions。

之后利用`globalATenDispatch`机制完成 Device 的 dispatch, dispatch 中注册的是`& VariableType::convolution`

（暂时不是很确定）CPU 下，其底层调用的是定义在`build/aten/src/ATen/TypeDefault.cpp`中的`TypeDefulat::convolution()`，其底层调用的是 `at::native::_convolution`

```c++
Tensor TypeDefault::convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
#ifdef BUILD_NAMEDTENSOR
    if (input.is_named() || weight.is_named() || bias.is_named()) {
        AT_ERROR("convolution: no named inference rule implemented.");
    }
#endif
    const OptionalDeviceGuard device_guard(device_of(input));
    return at::native::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
Tensor TypeDefault::_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
#ifdef BUILD_NAMEDTENSOR
    if (input.is_named() || weight.is_named() || bias.is_named()) {
        AT_ERROR("_convolution: no named inference rule implemented.");
    }
#endif
    const OptionalDeviceGuard device_guard(device_of(input));
    return at::native::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}
```

`at::native::convolution`定义在`aten/src/ATen/native/Convolution.cpp`

现在说明下 Python 端是如何逐步调用到 C++底层的`at::Tensor conv2d()`的，我们采用倒叙的方式，即 C++底层的`at::Tensor conv2d()`是如何被 Python 端逐步调用到的

1. `at::conv2d`会在`torch/csrc/autograd/generated/pytorch_torch_functions_dispatch.h`中被包含在`inline Tensor dispatch_conv2d`中

```c++
inline Tensor dispatch_conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {

  AutoNoGIL no_gil;
  return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
}
```

2. `dispatch_conv2d`会被`torch/csrc/autograd/generated/python_torch_functions.cpp`中的`PyObject * THPVariable_conv2d`做 C++ wrap

```c++
static PyObject * THPVariable_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, IntArrayRef[2] stride=1, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, int64_t groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_conv2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```

3. `THPVariable_conv2d`会在同文件`torch/csrc/autograd/generated/python_torch_functions.cpp`下，以 Python 名为`conv2d`存储在`static PyMethodDef torch_functions[]`

```c++
static PyMethodDef torch_functions[] = {
// ......
  {"conv2d", (PyCFunction)THPVariable_conv2d, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
// ......
```

4. `torch_functions`会在同文件下`torch/csrc/autograd/generated/python_torch_functions.cpp`，封装进`static PyTypeObject THPVariableFunctions`，成为`PyTypeObject THPVariableFunctions`的方法

```c++
static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableFunctions",         /* tp_name */
// ......
  torch_functions,                       /* tp_methods */
// ......
```

5. `THPVariableFunctions`会被在同一文件下的`torch/csrc/autograd/generated/python_torch_functions.cpp`函数`void initTorchFunctions(PyObject* module)`以`_VariableFunctions`为名字添加至`torch._C`中

```c++
void initTorchFunctions(PyObject* module) {
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(module, "_VariableFunctions", (PyObject*)&THPVariableFunctions) < 0) {
    throw python_error();
  }
}
```

6. `void initTorchFunctions`会在`torch/csrc/autograd/python_variable.cpp`中被`bool THPVariable_initModule(PyObject *module)`调用

```c++
bool THPVariable_initModule(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  return true;
}
```

7. `bool THPVariable_initModule`会在`torch/csrc/Module.cpp`中被`PyObject* initModule()`调用注册到`torch._C中`

# PyTorch 的 dispatch 的分发机制

首先需要明确，PyTorch 的分发机制存在两部分，

1. 首先，对于 device 的分发，确定该操作是在`enum class Backend { CPU, CUDA, HIP, SparseCPU, SparseCUDA, SparseHIP, MSNPU, XLA, QuantizedCPU, ComplexCPU, ComplexCUDA, Undefined, MkldnnCPU, NumOptions };`哪个**Backend**上运行的
2. 之后，对于 type 的分发，确定该操作是对 int, float 还是 xxx

v1.2.0 中对于 Tensor 的 op 是基于 ATen 的 dispather 实现的，目前开发者们正在逐步使用`c10 dispatcher`逐步取代 ATen 的 dispatcher，为什么这么做暂时我还没用弄明白

## ATen 的 Dispatcher

`ATenDispatch`会维护一个`std::unordered_map<std::string, ATenOpTable> op_tables_;`，每一个 op 对应一个 ATenOpTable。

`ATenOpTable`维护了两种 ops，一种是`VariableOp`，一种是`BaseOp`，在`getOp`时会根据`is_variable`来判断是获取哪种的 op，`getOp`返回的是函数指针

`getBaseOp`需要根据`Backend`的值来获取到对应 device 上的方法

`getVariableOp`则不需要根据`Backend`来判断

# 问题

## 动态链接库中的全局变量

**_References:_**

- [owent.net: C++又一坑:动态链接库中的全局变量](https://owent.net/2014/962.html)

知识点: 静态变量会在程序 startup 时（在 main 函数之前前）就会进行初始化

```

```

```

```
