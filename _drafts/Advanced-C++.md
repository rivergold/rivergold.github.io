# :fallen_leaf:Basics

## Distinguish Declare & Define

---

## Distinguish function & method

---

## Polymorphism

- 编译期多态: 模板
- 运行期多态: 动态绑定，主要通过虚函数实现

**_References:_**

- [CSDN: 多态性之编译期多态和运行期多态(C++版)](https://blog.csdn.net/dan15188387481/article/details/49667389)

## Virtual Function

子类 override 父类的虚函数

动态绑定： 声明基类的指针，利用该指针指向任意一个子类对象，调用相应的虚函数

**_References:_**

- [知乎: C++虚函数的作用是什么?](https://www.zhihu.com/question/23971699)

---

## `explicit`

This specifier specifies that a constructor doesn't allow `implicit conversions` or `copy-initialization`.<br>
**理解：** 该声明使得类的构造函数不允许隐式转换和拷贝初始化。

**_References:_**

- [cppreference: explicit specifier](http://en.cppreference.com/w/cpp/language/explicit)

---

## `static`

### static variable

### static function

### static data member

### static method

- 出现在类体外的方法定义不用写 statice 关键字
- 只能访问类中的 statice 成员
- 由于没有`this`指针的额外开销，静态方法比类的非静态方法要快一些

**_Ref:_** [知乎: C/C++ 中的 static 关键字](https://zhuanlan.zhihu.com/p/37439983)

**_References:_**

- &Delta; [CSDN: C/C++---static 函数，static 成员函数，static 变量，static 成员变量 再来理一理](https://blog.csdn.net/FreeApe/article/details/50979425)
- [CSDN: C++中 Static 作用和使用方法](https://blog.csdn.net/artechtor/article/details/2312766)

> static 成员函数里面不能访问非静态成员变量，也不能调用非静态成员函数

**_References:_**

- [CSDN: C/C++---static 函数，static 成员函数，static 变量，static 成员变量 再来理一理](https://blog.csdn.net/FreeApe/article/details/50979425)

---

## `final`

- [cppreference: final 说明符 (C++11 起)](https://zh.cppreference.com/w/cpp/language/final)

<!--  -->
<br>

---

<!--  -->

## Constructor Initializers List Must Be Uesed

- Members need to be initialized are `const` or references: because these types can only be initialized, they cannot be assigned.

- Members need to be initialized are of a class type that does not define a default constructor<br>

**理解：**

- 成员类型是没有默认构造函数的类。

- 需要初始化`const`成员或者`引用类型`的成员：因为这两种类型只能初始化，不能对其进行赋值。

<!--  -->
<br>

---

<!--  -->

## `protected` Members

a class uses `protected` for those members that it is willing to share with its `derived classes` but wants to protect from neneral access.

<!--  -->
<br>

---

<!--  -->

## `public`, `private` and `protected` Inheritance

Derivation can only inheriate `public` and `protected` members from **base**, but not `private`

- `public` inheritance: the **derivation** inheritate the **base** class's public and protected members, and their access control not change.

- `private` inheritance: all `public` and `protected` members in **base** are `private` in **derivation**

- `protected` inheritance: all `public` and `protected` members in **base** are `protected` in **derivation**

**_References:_**

- [tutorialspoint: C++ Inheritance](https://www.tutorialspoint.com/cplusplus/cpp_inheritance.htm)
- [blog: C++继承：公有，私有，保护](http://www.cnblogs.com/qlwy/archive/2011/08/25/2153584.html)

<!--  -->
<br>

---

<!--  -->

## Smart pointer

### How to release a `unique_ptr`

```c++
std::unique_ptr<int> up(new int(5));
// Release and delete
up.reset(nullptr)
```

**_References:_**

- [cppreference.com: std::unique_ptr::reset](https://en.cppreference.com/w/cpp/memory/unique_ptr/reset)
- [cppreference.com: how to delete unique_ptr](http://www.cplusplus.com/forum/general/119828/)

## lvalue and rvalue

**_References:_**

- :thumbsup:[简书: C++11 中的左值、右值和将亡值](https://www.jianshu.com/p/4538483a1d8a)
- :thumbsup:[知乎-专栏: C++右值引用](https://zhuanlan.zhihu.com/p/54050093)

An lvalue is an expression that refers to a memory location and allows us to take the address of that memory location via the & operator. An rvalue is an expression that is not an lvalue. Examples are

Ref [Thomas Becker's Homepage: C++ Rvalue References Explained](http://thbecker.net/articles/rvalue_references/section_01.html)

仔细品位[Move Semantics and Compiler Optimizations](http://thbecker.net/articles/rvalue_references/section_06.html)

<!--  -->
<br>

---

<!--  -->

## Copy Control

When a object of a class type is copied, moved, assigned, and destroyed, it need:

- copy constructor
- copy-assignment operator
- move constructor
- move-assignment operator
- desturctor

### Copy initialization

Happens:

- Pass an object as an argument to a parameter of nonreference type
  传递参数时，将 object 传递给非引用的参数
- Return an object from a function tha has a nonreference retrun type
  函数返回时，返回非引用类型的返沪值
- Brace initialize the element in an array or the members of an aggregate class
  大括号初始化数据、

Copy initialization ordinarily use the copy constructor. But if a class has a move constructor, then copy initialaztion sometimes use the move constuctor instead of the copy constructor.

<!--  -->
<br>

---

<!--  -->

## `NULL` & `nullptr`

**_Ref:_** [Quora: What's the difference between NULL and nullptr in C++?](https://www.quora.com/Whats-the-difference-between-NULL-and-nullptr-in-C++)

<!--  -->
<br>

---

<!--  -->

## Use `using namespace std` is bad

**_Ref:_** [stackoverflow: Why is “using namespace std;” considered bad practice?](https://stackoverflow.com/questions/1452721/why-is-using-namespace-std-considered-bad-practice)

<!--  -->
<br>

---

<br>
<!--  -->

# Class

## `friend`

**友元声明出现于类体内，并向一个函数或另一个类授予对包含友元声明的类的私有及受保护成员的访问权**

**_Ref:_** [cppreferences.com: 友元声明](https://zh.cppreference.com/w/cpp/language/friend)

简答的理解 e.g

```c++
class A{
    int data;
    friend B:B(char); // B是A的友元，B可以访问A的private和protected的成员
}
```

---

## Call base class constructor

```c++
#include <iostream>

class Parent {
public:
  int data;
};

class Child : public Parent {
public:
  Child(Parent const &rhs) : Parent(rhs){};
};

int main() {
  Parent parent = Parent();
  parent.data = 10;
  Child child = Child(parent);
  std::cout << child.data << std::endl;
}
```

**_References:_**

- [stackoverflow: What are the rules for calling the superclass constructor?](https://stackoverflow.com/questions/120876/what-are-the-rules-for-calling-the-superclass-constructor)

---

## Difference between `class` and `struct`

Only two differences:

- Default inherited permission: `class` is `private`, `struct` is `public`

- Default member access permission: `class` is `private`, `struct` is `public`

**_References:_**

- [CSDN: c++ struct 与 class 的区别](https://blog.csdn.net/hustyangju/article/details/24350175)

<!--  -->
<br>

---

<br>
<!--  -->

# STL

## vector

### Init vector size

```c++
vector<int> a;
a = vector<int>(10); // a size is 10
```

**_Ref:_** [stackoverflow: How to set initial size of std::vector?](https://stackoverflow.com/a/11457629)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Tricks

## Using `typedef` define a function pointer

```c++
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;
```

This code is from `caffe/tools/caffe.cpp`. Here, `BewFunction` is a function pointer of a function `int ()`

**_References:_**

- [Blog: [C++语法]关键字 typedef 用法](http://www.cnblogs.com/SweetDream/archive/2006/05/10/395921.html)

**注: C++ 11 下最好使用`using` 代替`typedef`**

<!--  -->
<br>

---

<!--  -->

## Calculate execution time

```c++
#include <chrono>

auto start = chrono::steady_clock::now();
// Code need to test time
auto end = chrono::steady_clock::now();
auto diff = end -start
// Print
cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;
// cout << chrono::duration <double, nano> (diff).count() << " ns" << endl;
```

Ref [stackoverflow: calculating execution time in c++](https://stackoverflow.com/a/47888078/4636081)

<!--  -->
<br>

---

<!--  -->

## PImpl

**Pointer to implementation**

**_Ref:_** [知乎: 如何写 C++代码，才能在封装成 Dll 的同时，自己程序包含的额外头文件不用加载进来？](https://www.zhihu.com/question/336227826/answer/757634065?hb_wx_block=0&utm_source=ZHShareTargetIDMore&utm_medium=social&utm_oi=37839882420224)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Tools

## fmt

- [Github](https://github.com/fmtlib/fmt)

Good format print for C++.

**_Ref:_** [知乎: iostream 是 C++ 的缺陷吗，为什么？](https://www.zhihu.com/question/24076731)

### Format String Syntax

**_Ref:_** [fmt doc: Format String Syntax](https://fmt.dev/latest/syntax.html)

<!--  -->
<br>

---

<!--  -->

# 临时记录

## `mutable`

**_References:_**

- [Blog: C++ 中的 mutable 关键字](https://liam.page/2017/05/25/the-mutable-keyword-in-Cxx/)

## `std::atomic`

> C++中对共享数据的存取在并发条件下可能会引起 data race 的 undifined 行为，需要限制并发程序以某种特定的顺序执行，有两种方式：使用 mutex 保护共享数据，原子操作：针对原子类型操作要不一步完成，要么不做，不可能出现操作一半被切换 CPU，这样防止由于多线程指令交叉执行带来的可能错误。非原子操作下，某个线程可能看见的是一个其它线程操作未完成的数据

**_References:_**

- [CSDN: C++并发实战 16: std::atomic 原子操作](https://blog.csdn.net/liuxuejiang158blog/article/details/17413149)

## `#pragma`

**_References:_**

- [CSDN: c++中#pragma 用法详解](https://blog.csdn.net/piaoxuezhong/article/details/58586014)

## `constexpr`

**C++11**

**_References:_**

- [cppreference.com: constexpr 说明符(C++11 起)](https://zh.cppreference.com/w/cpp/language/constexpr)

- [知乎: C++ const 和 constexpr 的区别？](https://www.zhihu.com/question/35614219/answer/63798713)

## `noexcept`

**C++11**

**_References:_**

- [cppreference.com: noexcept 运算符 (C++11 起)](https://zh.cppreference.com/w/cpp/language/noexcept)

## `std::is_name`

**_References:_**

- [cppreference.com: std::is_same](https://zh.cppreference.com/w/cpp/types/is_same)

## Class template

**_References:_**

- [cppreference.com: Class template](https://en.cppreference.com/w/cpp/language/class_template)

- [GeeksforGeeks: perm_identity Templates in C++](https://www.geeksforgeeks.org/templates-cpp/)

### Class template with multiple parameters

**_References:_**

- [GeeksforGeeks: Class template with multiple parameters](https://www.geeksforgeeks.org/class-template-multiple-parameters/)

### `typename` and `class`

现在的理解：在大部分的情况下，`typename`和`class`是一样的，但是最好对于类类型的声明用`class`，普通类型的声明用`typename`

**_References:_**

- [stackoverflow: Use 'class' or 'typename' for template parameters? [duplicate]](https://stackoverflow.com/a/213146/4636081)

- [Blog: C++ 中的 typename 及 class 关键字的区别](https://liam.page/2018/03/16/keywords-typename-and-class-in-Cxx/)

## `std::is_convertible`

**_References:_**

- [cppreference.com: std::is_convertible](https://en.cppreference.com/w/cpp/types/is_convertible)

## `std::move`

我目前的理解: 移交控制权

**_References:_**

- [cppreferences.com: 引用声明](https://zh.cppreference.com/w/cpp/language/reference)

- [知乎: 关于 C++右值及 std::move()的疑问？](https://www.zhihu.com/question/50652989)

## rhs: right hand side

**_References:_** [Cprogramming.com: rhs?](https://cboard.cprogramming.com/cplusplus-programming/34762-rhs.html)

## `shared_ptr` to stack object

**_References:_**

- [stackoverflow: c++ create shared_ptr to stack object](https://stackoverflow.com/questions/38855343/c-create-shared-ptr-to-stack-object)

---

## `shared_ptr`

**_References:_**

- [简书: C++11 智能指针](https://www.jianshu.com/p/e4919f1c3a28)

---

## 智能指针

目前的理解，智能指针是一个类，其对普通指针和引用计数进行了封装

**_References:_**

- [博客园: C++11 中智能指针的原理、使用、实现](https://www.cnblogs.com/wxquare/p/4759020.html)
- [博客园: C++ 引用计数技术及智能指针的简单实现](https://www.cnblogs.com/QG-whz/p/4777312.html)

### 侵入式智能指针

引用技术是放在 Object 的类里面，

侵入式智能指针的性能更好

> From PyTorch: intrusive_ptr<T> is an alternative to shared_ptr<T> that has better performance because it does the refcounting intrusively

**_References:_**

- :thumbsup::triangular_flag_on_post:[CSDN: C++侵入式智能指针的实现](https://blog.csdn.net/jiange_zh/article/details/52512337)

**理解**

PyTorch 中的`c10/util/intrusive_ptr.h`中的`intrusive_ptr_target`就是[CSDN: C++侵入式智能指针的实现](https://blog.csdn.net/jiange_zh/article/details/52512337)这里所说的:

> 1.将引用计数变量从资源类中抽离出来，封装成一个基类，该基类包含了引用计数变量。如果一个类想使用智能指针，则只需要继承自该基类即可；

`intrusive_ptr_target`是所有需要使用侵入式引用计数的类的基类，其内部有技术的变量

PyTorch 的核心 Tensor 是`aten/src/ATen/core.Tensor.h`中声明的`class CAFFE2_API Tensor`。其采用 Pimpl 形式（指向实现的指针），impl 为`impl_`，这是一个`c10::intrusive_ptr`,其指向的是`TensorImpl`，`TensorImpl`继承了`intrusive_ptr_target`

---

## `<functional>`

**_References:_**

- [Blog: C++ 11 STL | functional 标准库](https://www.sczyh30.com/posts/C-C/cpp-stl-functional/)

### `std::functional`

**_References:_**

- [CSDN: C++11 新特性之 std::function](https://blog.csdn.net/wangshubo1989/article/details/49134235)

---

## <numbers>

头文件 `<numbers>` 提供数个数学常数，例如 std::numbers::pi 或 std::numbers::sqrt2

**_References:_**

- [cppreference.com: 数值库](https://zh.cppreference.com/w/cpp/numeric)

---

## `inline`

**_References:_**

- [CSDN: c++ 内联函数（一看就懂）](https://blog.csdn.net/BjarneCpp/article/details/76044493)

---

## `reinterpret_cast`

**_References:_**

- [cppreference.com: reinterpret_cast 转换](https://zh.cppreference.com/w/cpp/language/reinterpret_cast)

---

## && after class method

```c++
template<class KernelFunctor, class... ConstructorParameters>
// enable_if: only enable it if KernelFunctor is actually a functor
guts::enable_if_t<guts::is_functor<KernelFunctor>::value, Options&&> kernel(TensorTypeId dispatch_key, ConstructorParameters&&... constructorParameters) && {
      static_assert(std::is_base_of<OperatorKernel, KernelFunctor>::value, "Tried to register a kernel functor using the kernel<Functor>() API, but it doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
      static_assert(std::is_constructible<KernelFunctor, ConstructorParameters...>::value, "Wrong argument list for constructor of kernel functor. The arguments to kernel<Functor>(arguments...) must match one of the constructors of Functor.");

      return std::move(*this).kernelFunctor<KernelFunctor, false>(dispatch_key, std::forward<ConstructorParameters>(constructorParameters)...);
    }
```

**_References:_**

- [知乎: c++成员函数声明()后加&或&&表示什么？](https://www.zhihu.com/question/47163104/answer/104614784)

---

## `std::enable_if`

**_References:_**

- [Blog: std::enable_if 的几种用法](https://yixinglu.gitlab.io/enable_if.html)

<!--  -->
<br>

---

<br>
<!--  -->

# AQ

## 什么时候需要使用`noexcept`?

TODO:

## `explicit`的作用是什么?

TODO:

**_References:_**

- [博客园: C++ 隐式类类型转换](https://www.cnblogs.com/QG-whz/p/4472566.html)

e.g.,

PyTorch 中对`implicit`的 constuctor 进行了注释

```c++
  // "Downcasts" a `Tensor` into a `Variable`. Only call this on tensors you
  // know are Variables.
  /*implicit*/ Variable(at::Tensor const& rhs) : at::Tensor(rhs) {
    TORCH_CHECK(
        is_variable() || !defined(),
        "Tensor that was converted to Variable was not actually a Variable");
  }
```

## `static_cast`

TODO:

## lambda 表达式的捕获列表

e.g. `Pytorch-aten/src/ATen/native/cpu/PowKernel.cpp`,

TODO:

- `[]`
- `[=]`
- `[&]`
- `[this]`
- ...

```c++
void pow_tensor_tensor_kernel(TensorIterator& iter) {
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "pow", [&]() {
      using Vec = Vec256<scalar_t>;
      cpu_kernel_vec(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return std::pow(base, exp);
        },
        [&](Vec base, Vec exp) -> Vec {
          return base.pow(exp);
        }
      );
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return std::pow(base, exp);
        }
      );
    });
  }
}
```

Here is an [example](https://github.com/rivergold/Cpp11/blob/master/cpp11/lambda_function.cc).

**_References:_**

- [简书: lambda 表达式 C++11](https://www.jianshu.com/p/923d11151027)
- [CSDN: C++11 中的 Lambda 表达式构成之——捕获值列表](https://blog.csdn.net/zh379835552/article/details/19542181)

---

## 指向函数指针的指针

E.g. from PyTorch `aten/src/ATen/native/DispatchStub.h`

```c++
template <typename rT, typename T, typename... Args>
// @rivergold: rT (*)(Args...) 是函数指针，但在DispatchStub定义时，typename rT 为函数指针
struct CAFFE2_API DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    if (device_type == DeviceType::CPU) {
      if (!cpu_dispatch_ptr) {
        cpu_dispatch_ptr = choose_cpu_impl();
      }
      return (*cpu_dispatch_ptr)(std::forward<ArgTypes>(args)...);
    } else if (device_type == DeviceType::CUDA) {
      AT_ASSERTM(cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
      return (*cuda_dispatch_ptr)(std::forward<ArgTypes>(args)...);
    } else if (device_type == DeviceType::HIP) {
      AT_ASSERTM(hip_dispatch_ptr, "DispatchStub: missing HIP kernel");
      return (*hip_dispatch_ptr)(std::forward<ArgTypes>(args)...);
    } else {
      AT_ERROR("DispatchStub: unsupported device type", device_type);
    }
  }

  // ...
}
```

### 函数指针

基本格式:

```c++
data_type (*func_pointer) (data_type arg1, data_type arg2, ..., data_type argn)
```

**_References:_**

- [RUNOOB.COM: C++ 函数指针 & 类成员函数指针](https://www.runoob.com/w3cnote/cpp-func-pointer.html)

---

## 模板特化

TODO:

**_References:_**

- [cppreference.com: 显式（全）模板特化](https://zh.cppreference.com/w/cpp/language/template_specialization)
- [Harttle Land Blog: C++模板的偏特化与全特化](https://harttle.land/2015/10/03/cpp-template.html)

### 类模板成员特化

E.g. from `aten/src/ATen/native/DispatchStub.h:147`

```c++
#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> decltype(fn) DispatchStub<decltype(fn), struct name>::arch = fn;
```

Here is an [example](https://github.com/rivergold/Cpp11/blob/master/cpp11/class_template_member_specialization.cc)

**_References:_**

- [CSDN: 特化类模板成员](https://blog.csdn.net/isscollege/article/details/75050179)
