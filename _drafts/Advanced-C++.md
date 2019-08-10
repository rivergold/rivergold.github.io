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
