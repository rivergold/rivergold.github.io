# Basics

## `explicit`

This specifier specifies that a constructor doesn't allow `implicit conversions` or `copy-initialization`.<br>
**理解：** 该声明使得类的构造函数不允许隐式转换和拷贝初始化。

***References:***
- [cppreference: explicit specifier](http://en.cppreference.com/w/cpp/language/explicit)

<!--  -->
<br>

***
<!--  -->

## Constructor Initializers List Must Be Uesed

- Members need to be initialized are `const` or references: because these types can only be initialized, they cannot be assigned.

- Members need to be initialized are of a class type that does not define a default constructor<br>

**理解：**

- 成员类型是没有默认构造函数的类。

- 需要初始化`const`成员或者`引用类型`的成员：因为这两种类型只能初始化，不能对其进行赋值。

<!--  -->
<br>

***
<!--  -->

## `protected` Members

a class uses `protected` for those members that it is willing to share with its `derived classes` but wants to protect from neneral access.

<!--  -->
<br>

***
<!--  -->

## `public`, `private` and `protected` Inheritance

Derivation can only inheriate `public` and `protected` members from **base**, but not `private`

- `public` inheritance: the **derivation** inheritate the **base** class's public and protected members, and their access control not change.

- `private` inheritance: all `public` and `protected` members in **base** are `private` in **derivation**

- `protected` inheritance: all `public` and `protected` members in **base** are `protected` in **derivation**

***References:***

- [tutorialspoint: C++ Inheritance](https://www.tutorialspoint.com/cplusplus/cpp_inheritance.htm)
- [blog: C++继承：公有，私有，保护](http://www.cnblogs.com/qlwy/archive/2011/08/25/2153584.html)

<!--  -->
<br>

***
<!--  -->

## Smart pointer

### How to release a `unique_ptr`

```c++
std::unique_ptr<int> up(new int(5));
// Release and delete
up.reset(nullptr)
```

***References:***

- [cppreference.com: std::unique_ptr::reset](https://en.cppreference.com/w/cpp/memory/unique_ptr/reset)
- [cppreference.com: how to delete unique_ptr](http://www.cplusplus.com/forum/general/119828/)

## lvalue and rvalue

An lvalue is an expression that refers to a memory location and allows us to take the address of that memory location via the & operator. An rvalue is an expression that is not an lvalue. Examples are

Ref [Thomas Becker's Homepage: C++ Rvalue References Explained](http://thbecker.net/articles/rvalue_references/section_01.html)

仔细品位[Move Semantics and Compiler Optimizations](http://thbecker.net/articles/rvalue_references/section_06.html)

<!--  -->
<br>

***
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
    传递参数时，将object传递给非引用的参数
- Return an object from a function tha has a nonreference retrun type
    函数返回时，返回非引用类型的返沪值
- Brace initialize the element in an array or the members of an aggregate class
    大括号初始化数据、

Copy initialization ordinarily use the copy constructor. But if a class has a move constructor, then copy initialaztion sometimes use the move constuctor instead of the copy constructor.


<!--  -->
<br>

***

<br>
<!--  -->

# Tricks

## Using `typedef` define a function pointer

```c++
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;
```

This code is from `caffe/tools/caffe.cpp`. Here, `BewFunction` is a function pointer of a function `int ()`

***References:***

- [Blog: [C++语法]关键字typedef用法](http://www.cnblogs.com/SweetDream/archive/2006/05/10/395921.html)

**注: C++ 11下最好使用`using` 代替`typedef`**

<!--  -->
<br>

***
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

***
<!--  -->