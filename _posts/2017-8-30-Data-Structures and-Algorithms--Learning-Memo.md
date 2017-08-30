# 基础
[演算法笔记](http://www.csie.ntnu.edu.tw/~u91029/index.html)
## 二进制
**`bin()`返回表示二进制的字符串**
: ```python
  a = 10
  bin(a)
  >>> '0b1010'
  ```

### 按位操作
**`&`按位与**
```python
3 & 5
>>> 1
```

**`|`按位或**
```python
3 & 5
>>> 7
bin(7)
>>> '0b111'
```

**`^`按位异或**
```python
2 ^ 5
>>> 7
```

**\~ 按位翻转**
```python
~3
>>> -4
# ~x = (x + 1) * -1
```

## 数字
**`float.is_integer()`判断一个float时候是个整数**
```python
(1.0).is_integer()
>>> True
```

## 二叉树
**基础知识**
- 一棵二叉树有n个元素，它就有n-1条边<br>
- 一棵二叉树的高度为h， 它最少有h个元素， 最多有$2^h - 1$个元素
    (注：等比数列前n项和：$\frac{a_1(1-q^n)}{1-q}$)
- 一棵二叉树有n个元素，它的高度最大为n，最小高度为$log_2(n+1)$

## 链表
### 常见题型
**检测链表中是否有环**<br>基本思想是，两个指针，一快一慢（一个步长为2，另一个为一），如果相遇，则有环

- 有环时检测环入口
    基本思想：在检测环的基础上，一指针从相遇点开始，另一指针从起点开始，步长均为1，再次相遇的地方便为环入口

***Reference***
- [面试精选:链表问题集锦](http://wuchong.me/blog/2014/03/25/interview-link-questions/)

## 数学
### 直线
直线方程
: $y = k_1x + b_2$

与其垂直的直线的直线方程
: $y = k_2x + b_2$, 其中：$k_1 * k_2 = -1$, 通过另外一个已知点再计算出$b_2$

### 夹角
笛卡尔坐标变换到极坐标
: 以笛卡尔坐标系的x轴正方向作为极坐标的正方向，对点计算$atan2(y, x) * 180 / \pi$
    <p align="center">
        <img src="http://i2.kiimg.com/586835/2d6b9b6bac258ab2.png" width="100%">
    </p>

向量间的夹角

: 向量$a$和$b$<br>
在不考虑旋转方向的话，默认夹角范围在$0 - \pi$, 使用
$$
a \cdot b = cons\theta * |a| |b|
$$

: 如果需考虑旋转方向(从向量a转到向量b)，有两种方法:
    - 使用$atan2(y, x)$计算两个向量与x轴正方向的夹角，之后相减
    - 利用
    $$
    a \times b = |a||b|sin\theta * \hat n
    $$
    其中$\theta$为a与b之间的夹角($0 - \pi$)<br>
    如果$a \times b > 0$，a -> b 为顺时针<br>
    反之，为逆时针

# 动态规划
**原理：** 找出问题所有可能的情况，从中找出最好的。但是手段相比于暴力搜索有了很大的变化，通过将问题拆分成母-子问题（不同的状态），并将之前的子问题（状态）的信息记录下来，便于其母问题（下一个状态）的求解。
思考的核心：
1. 母-子问题是什么
2. 从子问题的解如何得到母问题的解（状态转移方程是什么）

**动态规划和贪婪算法的区别是什么？** 贪婪算法每一步只考虑局部最优（当前情况下的最优解是什么），而动态规划会综合考虑之前的子问题，将其信息整合得出最优解。

***Reference***
- [CSDN：很特别的一个动态规划入门教程](http://blog.csdn.net/runninghui/article/details/8737969)

# Tricks
**排列与组合:**
- 排列
    ```python
    import itertools
    list(itertools.combinations('abc', 2))
    >>> [('a', 'b'), ('a', 'c'), ('b', 'c')]
    ```

- 组合
    ```python
    import itertools
    list(itertools.permutations('abc', 2))
    >>> [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
    ```

- 笛卡尔积
    ```python
    import itertools
    list(itertools.product('abc', 'xy'))
    >>> [('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y'), ('c', 'x'), ('c', 'y')]
    ```
<br>

**python2.7从控制台中获取输入:**
```python
import sys
while True:
    try:
        num = int(sys.stdin.readline())
        # all_data = set()
        # for i in range(num):
        #   all_data.add(int(sys.stdin.readline()))
        # all_data = sorted(all_data)
        # for i in range(len(all_data)):
        #     print all_data[i]
    except:
        break
```

**向set中添加元素:** 使用`.add()`
<br>

**计算最大公约数:**
<br>

**Python sorted:**
- `sorted(list, key=)`
- How to sorting list based on values from anotherlist?
    ```python
    a = list()
    b = list()
    # sort by a's value (from small to large)
    sorted_zip = sorted(zip(a, b), key=lambda x:(x[0]))
    # sort by a's value (from large to small)
    sorted_zip = sorted(zip(a, b), key=lambda x:(-x[0]))
    # sort by b's value (from small to large)
    sorted_zip = sorted(zip(a, b), key=lambda x:(-x[1]))
    # reverse zip
    x_sorted, y_sorted = zip(*sorted_zip)
    ```

- How to sort a dict?
    ```python
    import operator
    x = {'a': 2, 'b': 4, 'c': 1}
    sorted_x = sorted(x.items(),key=operator.itemgetter(1))
    >>> [('c', 1), ('a', 2), ('b', 4)]
    ```

**Count the occurrences of a list/string item in Python(如何计算list中某个元素出现的次数):**
```python
a = [1, 1, 2, 3, 1]
a.count(1)
>>> 3
# or
from collections import Counter
Counter(a)
>>> Counter({1: 3, 2: 1, 3: 1})
```
***Reference:***
- [stackoverflow: How can I count the occurrences of a list item in Python?](https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item-in-python)
- [stackoverflow: Difference between dynamic programming and greedy approach?](https://stackoverflow.com/questions/16690249/difference-between-dynamic-programming-and-greedy-approach)
<br>

**`bin()`函数，将整数转化为二进制字符串:**
```python
a = 1
bin(a)
>> '0b01'
# 最前面的0b表示二进制的意思
```

**灵活使用list切片**
```python
a = [1, 2, 3, 4]
b = a[::2]
print(b)
>> [1, 3]
# If you want to sum a[i * 2]
sum(a[::2])
>> 4
```

**python str.join()的使用方法:**

### Problems and Solution:**
- `sorted(list, reverse=True)`, will sort list from largest to smallest, but when you want to sort a list based on another list, it will change the order of equal element? How to solve it?
    ***Solution:*** <br> Using `key=lambda x: -x`.

    ***Reference:***
    - [Sorting list based on values from another list?](http://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list)
    - [How to sort (list/tuple) of lists/tuples?](http://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples)    - [What is the inverse function of zip in python? [duplicate]](http://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python)
