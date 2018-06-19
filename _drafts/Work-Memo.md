# Linux
## Linux查看硬盘分区
```
df -lh
```

***References:***
- [Linux 查看磁盘分区、文件系统、使用情况的命令和相关工具介绍](http://blog.51cto.com/13233/82677)

## Keep run wihout exist when using ssh to remote server
There are two solutions:
- Run with `nohup`
    ```bash
    nohup python file.py
    ```
- Run `tmux`, and using `tmux attach` to return to the same seesion
    ```bash
    tmux
    ```
    Then run what you want.

**Note:** Rivergold recommend the second method - using `tmux`

***References:***
- [stackExchange: How to keep a python script running when I close putty](https://unix.stackexchange.com/questions/362115/how-to-keep-a-python-script-running-when-i-close-putty)

## Count files num in a folder
- [Blog: ](https://blog.csdn.net/niguang09/article/details/6445778)

## Linux `tar`
- compress
    ```bash
    tar -czvf <file_name>.tar.gz <folder need compressed>
    tar -cjvf <file_name>.tar.bz2
    ```
- uncompress
    ```bash
    tar -xzvf <file_name>.tar.gz
    tar -xjvf <file_name>.tar.bz2
    ```

- `-c`: 建立一个压缩档案
- `-x`: 解压一个压缩档案
- `-z`: 是否具有`gzip`属性
- `-j`: 是否具有`bzip2`属性
- `-v`: 是否显示过程
- `-f`: 使用档名，需要参数的最后，后面立马接压缩包的名字

***References:***
- [Blog: tar压缩解压缩命令详解](https://www.cnblogs.com/jyaray/archive/2011/04/30/2033362.html)
- [Blog: linux tar (打包.压缩.解压缩)命令说明 | tar如何解压文件到指定的目录？](http://www.cnblogs.com/52linux/archive/2012/03/04/2379738.html)

## `Error mounting /dev/sdb1`
```bash
sudo ntfsfix /dev/sdb1 
```
***References:***
- [StackExchange: Error mounting /dev/sdb1 at /media/ on Ubuntu 14.04 LTS](https://askubuntu.com/questions/586308/error-mounting-dev-sdb1-at-media-on-ubuntu-14-04-lts)

## `ssh -X <user_name>@<ip>` occur error: `X11 forwarding request failed on channel 0
`
1. `sudo yum install xorg-x11-xauth`
2. Change `/etc/ssh/sshd_config`
    ```bash
    X11Forwarding yes
    X11UseLocalhost no
    ```
3. Reload ssh config
    ```bash
    sudo service sshd restart
    ```
4. Install `cmake-gui` to have a try

***References:***
- [stackoverflow: X11 forwarding request failed on channel 0](https://stackoverflow.com/questions/38961495/x11-forwarding-request-failed-on-channel-0)
- [Ask Xmodulo: How to fix “X11 forwarding request failed on channel 0”](http://ask.xmodulo.com/fix-broken-x11-forwarding-ssh.html)
- [StackExchange: ssh returns message “X11 forwarding request failed on channel 1”](https://unix.stackexchange.com/questions/111519/ssh-returns-message-x11-forwarding-request-failed-on-channel-1)

## Problems & Solutions
### `rm -rf *` recovery
- [blog: CentOS 恢复 rm -rf * 误删数据](https://my.oschina.net/coda/blog/222670)

## CentOS
### `rpm`的使用
- [Blog: CentOS RPM 包使用详解](http://blog.51cto.com/tengq/1930181)

<!--  -->
<br>

***
<!--  -->

# Tools
## Using `pip` install **OpenCV**
- [Medium: Installing OpenCV 3.3.0 on Ubuntu 16.04 LTS](https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961)


<!--  -->
<br>

***
<!--  -->


# Frameworks
## Torch
### Install
- [torch: Getting started with Torch](http://torch.ch/docs/getting-started.html)

When CentOS install torch, it occur error: **xxxx**

when install torch error: https://ubuntuforums.org/showthread.php?t=1670531

### Tensor slice index
```lua
M:sub(1, -1, 2, 3)
```
***References:***
- [Blog: torch Tensor学习：切片操作](https://www.cnblogs.com/YiXiaoZhou/p/6387769.html)

### Torch print dimension size of each layer
***References:***
- [Github torch/nn/Issues: How to print the output dimension size of each layer, just like "Top shape" in caffe](https://github.com/torch/nn/issues/922)

### Torch read weights of the model
```lua
model.modules[2].weight
```

***References:***
- [stackoverflow: [torch]how to read weights in nn model](https://stackoverflow.com/questions/32086106/torchhow-to-read-weights-in-nn-model)

### Problems and Solutions
- Install [torch-opencv](https://github.com/VisionLabs/torch-opencv) ouucurs errors: `Cannot Find OpenCV` or other errors about OpenCV
    **Solution:** Change `CMakeLists.txt` and set OpenCV path. Then use `luarocks make` to build and install package.
    ***References:*** 
    - [isionLabs/torch-opencv: Problem installing from source](https://github.com/VisionLabs/torch-opencv/issues/182)

## Lua
> Lua arrays begin with 1

### Getting input from the user in Lua
- [stackoverflow: Getting input from the user in Lua](https://stackoverflow.com/questions/12069109/getting-input-from-the-user-in-lua) 

## Deep Learning Model Convertor
- [Github: ysh329/deep-learning-model-convertor](https://github.com/ysh329/deep-learning-model-convertor)

# PyTorch
## PyTorch load torch model
```python
import torch
from torch.utils.serialization import load_lua
model = load_lua(<torch model>)
```

***References:***
- [PyTorch Forum: Convert/import Torch model to PyTorch](https://discuss.pytorch.org/t/convert-import-torch-model-to-pytorch/37)

# Tensorflow
## tf.nn, tf.layers, tf.contrib区别
***References:***
- [小鹏的专栏: tf API 研读1：tf.nn，tf.layers， tf.contrib概述](https://cloud.tencent.com/developer/article/1016697)

<!--  -->
<br>

***
<!--  -->

# Python
## Basics
### Insert element into list: `list.insert(<insert index>, element)`
```
a = [1,2,3]
a.insert(2, 0)
print(a)
>>> [1,2,0,3]
```

### Python multiprocessing
***References:***
- [Blog: Python多进程库multiprocessing中进程池Pool类的使用](https://blog.csdn.net/jinping_shi/article/details/52433867)
- [Blog: Python 多进程 multiprocessing.Pool类详解](https://blog.csdn.net/SeeTheWorld518/article/details/49639651)

When using `Pool` how to pass multiple arguments to function?
```python
from multiprocessing import Pool
def func_worker(arg_1, arg_2):
    <do what you want to>

pool = Pool(4) # Using 4 core
pool.starmap(func_worker, <data_need_processed>)
```

**Note:** Each element of <data_need_processed> is a list or tuple which contain both of `arg_1` and `arg_2`

E.G.
```python
row_data = pd.read_csv('./image_row_data.csv')
def download_worker(data_1, data_2):
    image_name = data_1
    url = data_2
    url = 'http://' + url
    logging.error(image_name)
    urllib.request.urlretrieve(url, './images/' + image_name)

pool = Pool(8)
download_row_data =  list(zip(row_data.loc[:, 'img_name'], row_data.loc[:, 'url']))
pool.starmap(download_worker, download_row_data)
```

***References:***
- [stackoverflow: Python multiprocessing pool.map for multiple arguments](https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments)
- [Rawidn's Blog: Python Pool.map多参数传递方法](https://www.rawidn.com/posts/Python-multiprocessing-for-multiple-arguments.html)
- [CODE Q&A: Python多处理pool.map多个参数](https://code.i-harness.com/zh-CN/q/530d5e)

### Logging
Basic using of `logging`
```python
import logging
# Config: output file, format, level
logging.basicConfig(filename=<outputfile.log>, format='%(message)s', level=logging.DEBUG)
logging.debug(<infor>)
logging.info(<infor>)
logging.warning(<infor>)
logging.error(<infor>)
logging.critical(<infor>)
```

level:
    - `DEBUG`
    - `INFO`
    - `WARNING`
    - `ERROR`
    - `CRITICAL`
`CRITICAL` is the highest level, when you set level as `CRITICAL`, other level information will not be print. The default level is `WARNING`.

`format` will set the format of the logging information format. The default format is `%(levelname)s:%(name)s:%(message)s`. Other format seting can be get from [Python doc-LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes) and [简书: python3 logging 学习笔记](https://www.jianshu.com/p/4993b49b6888)

***References:***
- [Python doc: Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [简书: python3 logging 学习笔记](https://www.jianshu.com/p/4993b49b6888)
- [python3-cookbook: ](http://python3-cookbook-personal.readthedocs.io/zh_CN/latest/c13/p11_add_logging_to_simple_scripts.html)

**Note:** By default, when you rerun python script, `logging` will not clear the pre `.log` file, it will add information at the end of the log file. If you want to clear the pre log file,
one solution is to checking the file exist first and remove it before logging print.
```python
import os
log_file_name = <file_name.log>
if os.path.exit(log_file_name): os.remove(log_file_name)
```

***References:***
- [stackoverflow: How to find if directory exists in Python]()(https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python)
- [stackoverflow: How to delete a file or folder](https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder)


### Download image from url

***References:***
- [stackoverflow: Downloading a picture via urllib and python](https://stackoverflow.com/questions/3042757/downloading-a-picture-via-urllib-and-python)
- [stackoverflow: AttributeError: 'module' object has no attribute 'urlretrieve'](https://stackoverflow.com/questions/17960942/attributeerror-module-object-has-no-attribute-urlretrieve)

Error `ConnectionResetError: [Errno 104] Connection reset by peer`: this error means that the request is so frequently, the server refuse some request. One solution is to do delay when download, here is an example:
```python
for url in urls:
    for i in range(10):
        try:
            r = requests.get(url).content
        except Exception, e:
            if i >= 9:
                do_some_log()
            else:
                time.sleep(0.5)
        else:
            time.sleep(0.1)
            break
     save_image(r)
```

***References:***
- [segmentfault: Python 频繁请求问题: [Errno 104] Connection reset by peer](https://segmentfault.com/a/1190000007480913)

### `zip`
- Combine two list:
    ```python
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = list(zip(a, b))
    print(c)
    >>> [(1, 4), (2, 5), (3, 6)]
    ```
    **理解:** `zip`就是将多个list对应位置上的element分别组合起来

- Unzip a list of list/tuple
    ```
    a = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    a1, a2, a3 = zip(*a)
    print(a1, a3, a3)
    >>> (1, 4, 7) (2, 5, 8), (3, 6, 9)
    ```

    **Note:** If you regard `a` as a matrix, `zip(*a)` will get each column of `a`.

***References:***
- [stackoverflow: How to unzip a list of tuples into individual lists? [duplicate]](https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists/12974504)
- [stackoverflow: What does the Star operator mean? [duplicate]](https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean)

### File Operations
#### Copy file
```python
import shutil
shutil.copy2(<from_path>, <to_path>)
```
E.G
```python
shutil.copy2('./test.jpg', '/data/test_dist.jpg')
```

***References:***
- [stackoverflow: How do I copy a file in python?](https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python)

#### Move file
```python
import os
import shutil
# Method 1:
os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# Method 2:
shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
```

***References:***
- [stackoverflow: How to move a file in Python](https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python)


### String
#### `"` in str
```python
a = '\"abc\"'
print(a)
>>> "abc"
```

#### `str.replace`
```python
<str>.replace(<old_str>, <new_str>)
```
You can use it to delete character by replace `<old_str>` as `''`

***References:***
- [stackoverflow: How to delete a character from a string using Python](https://stackoverflow.com/questions/3559559/how-to-delete-a-character-from-a-string-using-python)

### Issues
- [Global, Local and nonlocal Variables](https://www.python-course.eu/python3_global_vs_local_variables.php)

## Numpy
### Copy in numpy
***References:***
- [CSDN:【Python】numpy中的copy问题详解](https://blog.csdn.net/u010099080/article/details/59111207)

### `np.newaxis` add new axis to array
```python
x = np.array([1, 2, 3, 4])
x = x[:, np.newaxis]
print(x)
>>> array([[1],
           [2],
           [3],
           [4],
           [5]])
x = np.array(range(4)).reshape(2, 2)
x = x[..., np.newaxis]
print(x.shape)
>>> (2, 2, 1)
```

***References:***
- [stackoverflow: How does numpy.newaxis work and when to use it?](https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it)

### round number
***References:***
- [SciPy.org: numpy.around](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.around.html)

### Print full numpy array
```python
import numpy as np
np.set_printoptions(threshold=np.nan)
```

***References:***
- [stackoverflow: How to print the full NumPy array?](https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array)

### Change set a value for a region of array
```python
a = np.zeros((2, 2))
b = np.ones((2, 2))
b[0:2, 0:2] = a
```

***References:***
- [stackoverflow: Numpy array and change value regions](https://stackoverflow.com/questions/13000842/numpy-array-and-change-value-regions)

## Pandas
### Create `pd.DataFrame` with setting column names
```python
df = pd.DataFrame(<data>, columns=[name_1, name_2, ..., name_n])
```
### Rename DataFrame columns name
Using `df.rename(index=str, columns={<pre_name_1>: <new_name_1>, <pre_name_2>: <new_name_2>}`
```python
# Rename
df = df.rename(index=str, columns={'mask_coordinates': 'mask', 'resolution_processed': 'resolution'})
```

***References:***
- [pandas doc: pandas.DataFrame.rename](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html)

### `SettingWithCopyWarning`
- [SofaSOfa.io: pandas DataFrame中经常出现SettingWithCopyWarning](http://sofasofa.io/forum_main_post.php?postid=1001449)

### df.loc[]
If you want to using `df.loc[]` by index and column name, you should set a column named `index` of df
```
df['index'] = range(len(df))
df.set_index('index')
```
Then you can use like this:
```python
df.loc[:10, '<col_name>']
```

**Note:** `df.loc[:10]` will contain the index=10, the 11th element, it is different between Python.

## OpenCV
### When OpenCV read color image, color information is stored as BGR, in order to convert to RGB
```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(<img_path>) # BRG
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```

***References:***
- [PHYSIOPHILE: WHY OPENCV USES BGR (NOT RGB)](https://physiophile.wordpress.com/2017/01/12/why-opencv-uses-bgr-not-rgb/)

### When using OpenCV in IPython or Jupyter notebook, `cv2.imshow` and `cv2.waitKey(0)` cause crash.
Solution: Add `cv2.destroyAllWindows()`
```python
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

***References:***
- [Github Opencv/opencv Issues: opencv cv2.waitKey() do not work well with python idle or ipython](https://github.com/opencv/opencv/issues/6452)

## Anaconda
### When install Anaconda/Miniconda, occur error `bunzip2: command not found
You need to install `bzip2`
```bash
sudo apt-get install bzip2
```

<!--  -->
<br>

***
<!--  -->


# Other Tricks
## How to Generate ssh-key
```bash
ssh-keygen -t rsa -C "your_email@example.com"
```
- [Bitbucket Support: Creating SSH keys](https://confluence.atlassian.com/bitbucketserver/creating-ssh-keys-776639788.html)
- [Github Help: Connecting to GitHub with SSH](https://help.github.com/articles/connecting-to-github-with-ssh/)

## Using command install miniconda silent
```bash
bash <miniconda.sh path> -b -p <install path>
```
- [Conda doc: Installing on macOS](https://conda.io/docs/user-guide/install/macos.html#install-macos-silent)

## `pip` install OpenCV packages for Python
```bash
pip install opencv-python # If you need only main modules
pip install opencv-contrib-python # If you need both main and contrib modules
```

***References:***
- [PyPi: opencv-python 3.4.1.15](https://pypi.org/project/opencv-python/)

## Nvidia cuDNN download websites
- [Nvidia: cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

## Linux build OpenCV from source, occur cannot download `ippicv`
**Solution:** Manually download `ippicv` from [here](https://raw.githubusercontent.com/Itseez/opencv_3rdparty/81a676001ca8075ada498583e4166079e5744668/ippicv/ippicv_linux_20151201.tgz)
And move it into `opencv-3.1.0/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e`

***References:***
- [Blog: Cenots7编译Opencv3.1错误：下载ippicv，解决方案](https://blog.csdn.net/daunxx/article/details/50495111)
