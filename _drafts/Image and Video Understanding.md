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
**ROI Pooling:** 将大小为$(h_{roi}, w_{roi})$的ROI所对应的feature map，大小为$(h_{f}, w_{f})$分成$H * W$个sub-windows，每个sub-window的大小为$(h_{f}/H, w_f/W)$. 

> Pooling is applied independently to each feature map channel, as in standard max pooling.

该方法是从SSPNet的多尺度的空间金字塔池化简化而来的。

对于不同输入图片，分成的grid个数要相同，但grid的大小可变

***References***
- [知乎专栏：晓雷机器学习笔记 SPPNet-引入空间金字塔池化改进RCNN](https://zhuanlan.zhihu.com/p/24774302)
- [deepsense.ai: Region of interest pooling explained](https://blog.deepsense.ai/region-of-interest-pooling-explained/)