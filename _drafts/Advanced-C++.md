# :fallen_leaf:Basics

## `explicit`

This specifier specifies that a constructor doesn't allow `implicit conversions` or `copy-initialization`.<br>
**理解：** 该声明使得类的构造函数不允许隐式转换和拷贝初始化。

**_References:_**

- [cppreference: explicit specifier](http://en.cppreference.com/w/cpp/language/explicit)

---

## `static`

**_Ref:_** [知乎: C/C++ 中的 static 关键字](https://zhuanlan.zhihu.com/p/37439983)

**_References:_**

- &Delta; [CSDN: C/C++---static 函数，static 成员函数，static 变量，static 成员变量 再来理一理](https://blog.csdn.net/FreeApe/article/details/50979425)
- [CSDN: C++中 Static 作用和使用方法](https://blog.csdn.net/artechtor/article/details/2312766)

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

## rhs: right hand side

**_References:_** [Cprogramming.com: rhs?](https://cboard.cprogramming.com/cplusplus-programming/34762-rhs.html)

## `shared_ptr` to stack object

**_References:_**

- [stackoverflow: c++ create shared_ptr to stack object](https://stackoverflow.com/questions/38855343/c-create-shared-ptr-to-stack-object)
