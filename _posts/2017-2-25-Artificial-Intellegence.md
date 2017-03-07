<!-- # *Artificial Intellegence* -->
This file is for collecting and recording good materials about artificial intellegence, include deep learning and  reinforcement learning.

<br>
## Valuable Websites
- [GitXiv](http://www.gitxiv.com)<br>
收集了与计算机相关的很多论文与github的源代码


- [Deep Learning Papers Reading Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)<br>
非常值得收藏的github，包括了很多最重要的深度学习的论文
知乎上还有其[翻译的版本](https://zhuanlan.zhihu.com/p/25549497)

- [AI突破性论文及代码实现汇总](https://zhuanlan.zhihu.com/p/25191377)<br>
汇总了关于AI的一些很重要的论文与网站资源

- [Data Science Python notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)<br>
包含了与数据科学有关的很多工具的python使用教程

- [Notes on Data Science, Machine Learning, & Artificial Intelligence](http://chrisalbon.com/)<br>
很不错的有关数据科学、机器学习和人工智能的笔记
<br>

## Awesome Gtihub
- [SSD: Single Shot MultiBox Detector ](https://github.com/weiliu89/caffe/tree/ssd)
- [Upvote Semantic Perceptual Image Compression using Deep Convolution Networks](https://github.com/iamaaditya/image-compression-cnn)
- [
Aerial Informatics and Robotics Platform](https://github.com/Microsoft/AirSim)
- [DeepPose](http://www.gitxiv.com/posts/JpkGNjWzWQM2fF7Dn/deeppose)
- [DeepLab: Semantic Image Segmentation with Deep CNNs, Atrous Convolution, and Fully Connected CRFs](http://www.gitxiv.com/posts/zbiNHqMHmpxFFNYZm/deeplab-semantic-image-segmentation-with-deep-cnns-atrous)

## Worthy Paper
- Human pose
    - [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release)<br>
    CVPR 2016, pose estimation based on single color image. Use caffe, python/matlab
    - [DeeperCut Part Detectors](https://github.com/eldar/deepcut-cnn)<br>
    ECCV 2016, CNN-based body part detectors. Use caffe and python.
    - [Stacked Hourglass Networks for Human Pose Estimation](http://www-personal.umich.edu/~alnewell/pose/)<br>
    ECCV 2016, Using Torch and Lua
    - [Human pose estimation via Convolutional Part Heatmap Regression](https://www.adrianbulat.com/human-pose-estimation)<br>
    ECCV 2016. CNN and Heat-map. Use Lua and Torch


## Basic Knowledge

### Understanding of hyperparameters(超参数) in machine learning
- cannot be directly learned from the regular training process
- hyperparameters are ususlly fixed before the actural training process begins
<https://www.quora.com/What-are-hyperparameters-in-machine-learning><br>
对于算法的选择也是一种超参数

### How to install Tensorflow  
Tensorflow官网  <https://github.com/tensorflow/tensorflow>  
Problem during installation of cuda:  
- How to run .run file  
<http://askubuntu.com/questions/18747/how-do-i-install-run-files>

- install cuda on ubuntu16.04参考
<https://www.pugetsystems.com/labs/hpc/Install-Ubuntu-16-04-or-14-04-and-CUDA-8-and-7-5-for-NVIDIA-Pascal-GPU-825/>

- Install/build Tensorflow from source
之前由于tensorflow安装文件不支持cuda8.0， 所有就自己用源码编译了tensorflow
  - how to install bazel:
    https://www.bazel.io/versions/master/docs/install.html

  - build tensorflow from source:
    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#installing-from-sources>

- 解决安装tenforflow出现的setup-tool错误
http://www.peacesky.cn/post/%E8%A7%A3%E5%86%B3%E5%AE%89%E8%A3%85Tensorflow%E6%97%B6%E7%9A%84setup-tool%E9%94%99%E8%AF%AF<br>

### Very good practice for Tensorflow
http://blog.topspeedsnail.com/archives/tag/tensorflow/page/3<br>

### Good way to understand Backpropagation
<http://neuralnetworksanddeeplearning.com/chap2.html>

### Good Recurrent Neural Network Tutorial
- Awesome Recurrent Neural Networks (from Github)   
<https://github.com/kjw0612/awesome-rnn>
- A blog from 'WILDML'  
<http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/>
<br>

### Good source codes from github(most have paper and source code)
- [iGAN](https://github.com/junyanz/iGAN) by Jun-Yan Zhu, Berkeley


<!-- =============================================================== -->
<!--                         Keras                          -->
<!-- =============================================================== -->
<br>
## Keras
<!-- ## <p align="center">********************** <br> Keras <br> **********************</p> -->

### RNN in keras
Understand how to use `return_sequences=True/False`  
<https://github.com/fchollet/keras/issues/2496><br>
<https://github.com/fchollet/keras/issues/1360>

###Good way to use RNN for classification/regression  
Understand what is (n_samples, timesteps, X_dimension)  
<https://github.com/fchollet/keras/issues/1904>

### How to build many to many model of RNN in keras
- problem  
<https://github.com/fchollet/keras/issues/2403>  
- solution  
<https://github.com/fchollet/keras/issues/562>

### Keras LSTM 的基本使用example
Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras  
<http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/>

### How to connect a dense to a LSTM
针对于RNN中 many-to-many 的模型，如何将其连接做分类？使用TimeDistributed，解决方法如下
<https://github.com/fchollet/keras/issues/915><br>

### Solve sequences of different length in RNN/LSTM
- [Is padding necessary for LSTM network?](https://github.com/fchollet/keras/issues/2375)

### Keras model are trained on Numpy arrays of input data and labels.

> 注意网络的输入与输出

```python
model = Sequential()
model.add(Dense(1, input_dim=784, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 1))

# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)
```

```python
# for a multi-input model with 10 classes:

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
from keras.utils.np_utils import to_categorical
data_1 = np.random.random((1000, 784))
data_2 = np.random.random((1000, 784))

# these are integers between 0 and 9
labels = np.random.randint(10, size=(1000, 1))
# we convert the labels to a binary matrix of size (1000, 10)
# for use with categorical_crossentropy
labels = to_categorical(labels, 10)

# train the model
# note that we are passing a list of Numpy arrays as training data
# since the model has 2 inputs
model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32)
```

<br>

<!-- =============================================================== -->
<!--                     Learning Sources                  -->
<!-- =============================================================== -->
<br>
## Leaning Sources

### Pre-trained Neural Networks
[Gradientzoo](https://www.gradientzoo.com/)<br>
[fchollet/deep-learning-models](https://github.com/fchollet/deep-learning-models)<br>

#### 网络介绍
- [GoogleLeNet](http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html)

### CS231n
<http://cs231n.stanford.edu/syllabus.html><br>

小知识点：
- 目前处理的图片都必须是同样大小的
- 主流的方式是使用方形的图片

### 个人理解

- 机器学习，是个学习+推断的过程，从数据中学习到的是经验，当进行预测时，系统的输入就是你所获得的证据

- 线性分类器类似于模版匹配(对哪一个地方相应很高)，只是学习到的模版类型比较单一
  神经网络可以学到很多模版

- loss function的作用：衡量分类/回归结果的好坏。从而优化它，找到最优的参数获得最好的分类/回归的性能

- 多分类支持向量机, loss function  
$$
L_i=\sum_{j\ne y_i }{max(0, s_j - s_{y_i} + 1)}
$$

- 为什么使用正则化项？
  - 当有多组参数/权重(w)获得了同样的结果(最小化了loss function)时，我们想要寻出最符合我们要求的
<br><br>

- L2正则化项，使模型尽可能多的利用每一纬的输入

- softmax classifier是logistic的一般化形式
  - softmax functionis  
  $$
  \begin{array}{c}
    f = \frac {e^{s_k}} {\sum_{j}{e^{s_j}}} \\ \\
    s=f(x_i; w)
  \end{array}
  $$
- 步长(learning rate)是一个比较头疼的超参数(选多少？)
- mini batch
针对于每一次参数的update，不会去计算所有的训练集样本的损失函数，只是从其中取出一部分：Mini-batch Gradient Descent
- 机器学习中的优化问题(凸优化问题)：想象为在山谷中找到最低的点
- sift/HOG特征：统计出边缘的方向的分布图
- 特征字典：真的就有点像python中的dictionary，{每种特征：统计结果}
- 将数学表达式转换为图(graph)表示

<!-- <p align="center">
  <img src="../imgs/softmax_cs231n.png" alt="Drawing;" style="width: 600px;"/>
</p> -->
