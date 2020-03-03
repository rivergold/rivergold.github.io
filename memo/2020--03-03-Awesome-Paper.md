---
title: "Awesome Paper"
last_modified_at: 2020-03-03
categories:
  - Paper
tags:
  - Awesome
  - Paper
---

A collection of awesome paper, major in face detection, face alignment, image inpainting.

## :fallen_leaf:Face Detection

### [(arxiv-2019) RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)

#### Source Code

- [Offical Code](https://github.com/deepinsight/insightface/tree/master/RetinaFace): Use mxnet
- [supernotman/RetinaFace_Pytorch](https://github.com/supernotman/RetinaFace_Pytorch)
- [Tencent/ncnn](https://github.com/Tencent/ncnn/blob/master/examples/retinaface.cpp)

### [(arxiv-2019) CenterFace: Joint Face Detection and Alignment Using Face as Point](https://arxiv.org/abs/1911.03599)

#### Source Code

- [Offical Code](https://github.com/Star-Clouds/CenterFace): No training code, support many platform format models.

## :fallen_leaf:Image Inpainting

### [(AAAI-2020) Learning to Incorporate Structure Knowledge for Image Inpainting](https://arxiv.org/abs/2002.04170)

基于多任务学习，将图片的结构化信息（边缘、梯度）作为 Ground Truth 一起训练来指导图像修复效果

- 处理图片
- 看效果支持矩形和非矩形的 mask
- 没有训练代码，仅有推理代码: [Github: YoungGod/sturcture-inpainting](https://github.com/YoungGod/sturcture-inpainting)

### [(AAAI-2020) Region Normalization for Image Inpainting](https://arxiv.org/abs/1911.10375)

针对于归一化做了修改，不同区域（修复区域和非修复区域）采用不同的均值和方差处理

- 处理图片
- 看效果支持矩形和非矩形的 mask
- 代码暂未开源: [Github: geekyutao/RN](https://github.com/geekyutao/RN)

### [(ICCV-2019) Copy-and-Paste Networks for Deep Video Inpainting](https://arxiv.org/abs/1908.11587)

从相邻帧中找到上下文信息来修复当前帧的待修复区域，并且添加了一个 alignment 网络来计算当前帧与相邻帧的接近程度，从而从更多的相邻帧中获取有效信息

- 处理视频
- [Demo](https://www.youtube.com/watch?v=bxerAkHAttE&feature=youtu.be)效果看起来可以
- 速度 4fps 左右
- 没有训练代码，只有推理代码: [Github: shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting](https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting)

### [(BMVC-2019) Learnable Gated Temporal Shift Module for Deep Video Inpainting](https://arxiv.org/abs/1907.01131)

在下一篇的基础之上 Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN，采用 temporal shift module (TSM) for action recognition（暂未了解）的方法，针对于之前方法参数多、训练推理慢的问题进行优化

- 处理视频
- 处理速度 80fps（具体测试环境暂未确定）
- [Demo](https://www.youtube.com/watch?v=87Vh1HDBjD0&list=PLPoVtv-xp_dL5uckIzz1PKwNjg1yI0I94&index=32&t=0s)效果看起来可以([完整的 Demo](https://drive.google.com/drive/folders/1zMRqkDsv2X2BZ3lygRX3TiAc16qmCkJO))
- 有训练代码: [Github: amjltc295/Free-Form-Video-Inpainting](https://github.com/amjltc295/Free-Form-Video-Inpainting)

### [(ICCV-2019) Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN](https://arxiv.org/abs/1904.10247)

基于 3D 门卷积和时序 PatchGAN 结合，搜集了 FVI 数据集（28,000 free-form mask videos）

- 处理视频
- 处理速度 30fps（具体测试环境暂未确定）
- 和上一篇是同一个作者
- free-form mask 中含有矩形框的 mask
- [Demo](http://bit.ly/2Fu1n6b)效果看起来还行，在上一篇中有对于实验
- 有训练代码: [Github: amjltc295/Free-Form-Video-Inpainting](https://github.com/amjltc295/Free-Form-Video-Inpainting)，和上一篇是同一个仓库

### [(CVPR-2019) Deep-Video-Inpainting](https://arxiv.org/abs/1905.01639)

基于 image inpainting 的 encode-decoder 网络，结合 LSTM 建模时许特征，采用 flow loss 和 warp loss 进行训练

- 处理视频
- 12fps 在 256x256 的图片上
- [Demo](https://www.youtube.com/watch?time_continue=9&v=RtThGNTvkjY&feature=emb_logo)效果很好
- 有训练代码: [Github: mcahny/Deep-Video-Inpainting](https://github.com/mcahny/Deep-Video-Inpainting)

### [(CVPR-2019) Deep Blind Video Decaptioning by Temporal Aggregation and Recurrence](https://sites.google.com/view/bvdnet/)

- 上一篇论文的作者
- 针对于去字幕
- 问题：论文中使用的数据图片为 128x128 大小
- 没有代码

### [(CVPR-2019) Deep Flow-Guided Video Inpainting](https://arxiv.org/abs/1905.02884)

基于光流，先估计光流之后结合 warp 和 image inpainting 的方法（DeepFill v1）进行修复

- 处理视频
- 有训练代码: [Github: nbei/Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting)

### [(CVPR-2018) Generative Image Inpainting with Contextual Attention](https://arxiv.org/pdf/1801.07892.pdf)

Deepfill v1，基于 GAN 和 Context convolution。去台标项目刚开始时用的第一个深度学习方法。

- 处理图片
- 在 Place2 数据集（风景数据集）上训练了模型
- 论文里说在 1080Ti 上 0.2s/frame; 去 logo 项目中采用 crop 的方式，由于 logo 区域较小速度有所提升，印象中为 50ms 左右

### [(ICCV-2019) SC-FEGAN: Face Editing Generative Adversarial Network with User’s Sketch and Color](https://arxiv.org/pdf/1902.06838.pdf)

- 论文针对于人脸，但是 baseline 模型中可以处理其他类型的图片
- 测试处理速度 60ms/frame
