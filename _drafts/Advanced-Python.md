# :fallen_leaf: New

## Python 3.8

**_References:_**

- [Python 开发者: Python 3.8 新功能大揭秘](https://mp.weixin.qq.com/s?__biz=MzA4MjEyNTA5Mw==&mid=2652569675&idx=1&sn=928715a62ad52aebd66b1076c6b7b4ff&chksm=84652801b312a1170c2beb98b25431353e5dec43c346390f2bf2ce91256c0313d69c98a15c3a&mpshare=1&scene=1&srcid=#rd)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Class

## metaclass

**_Ref:_** [Python 之旅: 陌生的 metaclass](http://funhacks.net/explore-python/)

### What is metaclass, class and instance?

```shell
类是实例对象的模板，元类是类的模板
+----------+             +----------+             +----------+
|          |             |          |             |          |
|          | instance of |          | instance of |          |
| instance +------------>+  class   +------------>+ metaclass|
|          |             |          |             |          |
|          |             |          |             |          |
+----------+             +----------+             +----------+
```

### Use of metaclass

```python
class Foo(metaclass=PrefixMetaclass):
    name = 'foo'
    def bar(self):
        print('bar')
```

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Decorator

**我的理解**:

> 装饰器是可调用的对象，其参数是另一个函数（被装饰的函数）。装饰器可能会处理被装饰的函数，然后把他返回，或者将其替换成另一个函数或可调用对象。
> 严格来说，装饰器只是语法糖。
> 装饰器的一大特性是，能把被装饰的函数替换成其他函数。第二特性是，装饰器在加载模块时立即执行。

## How to use

**_References_**

- [segmentfault: Python 装饰器使用指南](https://segmentfault.com/a/1190000010681026)
- [简书: Python - 学装饰器之前，有几个点要理解](https://www.jianshu.com/p/1369d610f8bb)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Awesome Tricks

## Make Class like a `dict`

Can use `[]` or `get` to get attribute in the class.

**Good example: [Github open-mmlab: mmcv/mmcv/utils/config.py](https://github.com/open-mmlab/mmcv/blob/9097957434da1308739c03aff53ac147f1500c56/mmcv/utils/config.py#L142)**

<!--  -->
<br>

---

<!--  -->

## Set random seed

```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**_Ref:_** [Github open-mmlab/mmdetection: mmdet/apis/env.py](https://github.com/open-mmlab/mmdetection/blob/58c415a069ceab37bab76c47da72824f4181cff6/mmdet/apis/env.py#L53)

<!--  -->
<br>

---

<!--  -->

## Convert str into class object

```python
a = 'pow'
b = eval(a)(2, 3)
print(b)
>>> 8
```

**_Ref:_** [stackoverflow: Convert string to Python class object?](https://stackoverflow.com/a/1178089)

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Knowledge Points

## Variable Scope

**_Ref:_** [CSDN: Python 变量作用域](https://blog.csdn.net/cc7756789w/article/details/46635383)

There are four type scope in Python:

- L (Local) 局部作用域
- E (Enclosing) 闭包函数外的函数中
- G (Global) 全局作用域
- B (Built-in) 内建作用域

Python find variable in order of `L -> E -> G -> B`.

Only `def`, `class` and `lambda` will change variable scope. Other like `if/elif/else`, `try/except`, `for/while`, `with` will not change variable scope.

<!--  -->
<br>

---

<br>
<!--  -->

# :fallen_leaf:Anaconda

## Set `ananconda/bin` in `~/.zshrc`

Edit `~/.zshrc`:

```shell
export PATH=~/software/anaconda/bin:$PATH
```

When use a new env which is different from the base, it's better to create new env for Python. Then a new folder `${ANANCONDA}/envs/<env_name>/` will created.

When install some Python package, it will also install some third party libs which will be added into `ananconda/bin`. Because of `export PATH=~/software/anaconda/bin:$PATH`, `anaconda/bin` will takes precedence over `usr/bin`. So when you want to install a package via Anaconda and it maybe conflict with system libs, you should create a new conda env.

**E.g.**

When you want to use Python with FFMPEG, `pip install opencv-python` does not with FFMPEG. `conda install opencv` has FFMPEG, and it will install it in `anaconda/bin`, but this FFMPEG not build with X264 (--disable-libx264). So you shuold create a new env for this to avoid influencing the system env.
