# Datasets
- [Github: video understanding datasets](https://github.com/yoosan/video-understanding-dataset)

# Tensorflow
- [TFLearn](http://tflearn.org/)
- [Tensorflow中文社区](http://www.tensorfly.cn/)

# Semantic Segmentation
- [Qure.ai Blog: A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)

# Object Detection
- [知乎：如何评价rcnn、fast-rcnn和faster-rcnn这一系列方法？](https://www.zhihu.com/question/35887527)

# Awesome Home Page
- [Liang-Chieh Chen's home page](http://liangchiehchen.com/)
- [Ross Girshick(rbg)'s home page](http://www.rossgirshick.info/)
- [知乎专栏：晓雷机器学习笔记](https://zhuanlan.zhihu.com/xiaoleimlnote)

# Deep Learning
- [Caleb的机器学习工厂：理解Deconvolution（反卷积）](http://calebml.leanote.com/post/%E7%90%86%E8%A7%A3deconvolution%EF%BC%88%E5%8F%8D%E5%8D%B7%E7%A7%AF%EF%BC%89)
- [StackExchange: What are deconvolutional layers](https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers)
- [Towards Data Science: Up-sampling with Tranposed Convolution](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)

# Read Papers Notes
## Mask R-CNN
以Fatser R-CNN为基础，使用FCN添加了并行的segmentation mask

## Fast R-CNN
**ROI Pooling:** ROI指的是在feature map上的矩形框
> In this paper, an RoI is a rectangular window into a conv feature map.

将在feature map上的大小为$(h_{roi}, w_{roi})$的ROI所对应的feature map分成$H * W$个sub-windows，每个sub-window的大小为$(h_{f}/H, w_f/W)$。其中，每个sub-window称为RoI bin，而$H, W为下一层输入所需要的大小。 

> Pooling is applied independently to each feature map channel, as in standard max pooling.

该方法是从SSPNet的多尺度的空间金字塔池化简化而来的。

对于不同输入图片，分成的grid个数要相同，但grid的大小可变

***References***
- [知乎专栏：晓雷机器学习笔记 SPPNet-引入空间金字塔池化改进RCNN](https://zhuanlan.zhihu.com/p/24774302)
- [deepsense.ai: Region of interest pooling explained](https://blog.deepsense.ai/region-of-interest-pooling-explained/)

## Faster R-CNN
核心思想：使用Region Proposal Network(RPN)替代Fast R-CNN中的selective search方法生成region proposal，之后再将Fast R-CNN接在RPN之后进行分类和检测窗位置的回归。

RPN和之后的Fast R-CNN共享了位于前端的网络层，用于获得shared feature maps

RPN:由$3*3$的卷积和$1*1$的卷积构成(即使用FCN实现)
训练RPN
将原始图像的短边rescale为600， 输入单张图像，在shared feature map的基础上对每个像素点都生成k个anchor（带有锚点的box），$k=(n*m*9)$, n, m为feature map的大小，共有3种大小的anchor以及3种scale ratio故有9种anchor，并计算生成的anchor与数据集中标记的box的IoU，来实现正负样本的标记（2类，有无object），从而训练网络。
- Net: VGG / Resnet
- input: 原始图像
- label:
    - cls: (h, w, num_anchors * 2)
    - bounding box: (h, w, num_anchors * 4)

**RPN的作用就是从anchor中挑选出合适的作为proposal**

Ross Girshick使用caffe编写的Faster R-CNN的代码中，没有事先处理、生成好Label，而是在训练过程中在线计算label，之后计算误差。

***References:***
- [知乎专栏：晓雷机器学习笔记 Faster R-CNN](https://zhuanlan.zhihu.com/p/24916624)
- [Medium: Faster R-CNN Explained](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8)

## Mask R-CNN
Instance Segmentation has two sub-problems
- Object Detection -> bounding box
- Semantic Segmentation -> shaded masks