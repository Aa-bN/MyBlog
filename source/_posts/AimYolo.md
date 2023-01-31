---
title: AimYolo
date: 2023-01-30 16:58:05
author: 好想摸鱼
img: /medias/BlogPictures/1. AimYolo/cover/cover.PNG
top: True
cover: False
coverImg: https://wsblog.netlify.app/medias/BlogPictures/1. AimYolo/cover/cover.PNG
password:
toc: true
mathjax: true
summary: AI外挂——基于YOLOv5的射击类游戏瞄准辅助。An AI plug-in - targeting aid for shooting games based on YOLOv5.
categories: demo
tags:
    - AI
    - 深度学习
    - YOLOv5
    - 目标检测算法
    - FPS
    - 外挂
keywords:
    - AI
    - 深度学习
    - YOLOv5
    - 目标检测算法
    - FPS
    - 外挂
reprintPolicy: cc_by
---

本文使用**YOLOv5-2.0**和**PyTorch**，实现了一款基于目标检测算法的射击类游戏瞄准辅助，俗称“AI外挂”。（文中含项目地址及演示视频）

---

## 一、前言
本项目出于个人兴趣而写，为demo级别，较为粗糙，优化空间很大。灵感来源于B站up主“林亦LYi”的视频。

1. [**项目地址**](https://github.com/Aa-bN/AimYolo)  
   - 已在GitHub开源
   ---
2. **项目结构**   
    - PyTorch，[机器学习框架](https://pytorch.org/)
    - YOLOv5-2.0，[项目地址](https://github.com/ultralytics/yolov5)
    - 自瞄模块 
   ---
3. **注意事项**  
    - *本项目旨在激起深度学习、游戏安全等方面爱好者的兴趣，仅供学习交流
    - *切勿用于盈利及违法用途，以及进行任何破坏游戏公平的行为
    - 有封号风险，推荐断网使用
    - 推荐在Anaconda中部署并使用
    - 10系显卡效果明显不如30系（o(╥﹏╥)o）
    ---

值得注意的是，AI外挂几年前就已经出现，最初为YouTube上的一位创作者开发，网络上也可以找到一些版本。本项目理论上可以用于CSGO、APEX、PUBG等多种游戏，实际以CSGO为例进行了模型的开发与使用。相对于传统外挂，AI外挂多基于目标检测算法，不会修改游戏的本地内存，也不会上传恶意数据。本文后续有6个部分：**开发过程**、**模型开发**、**核心代码**、**项目部署与展示**、**总结**与**参考**。

---

## 二、开发过程
本节分为**总体设计**、**屏幕实时捕获**、**鼠标定位与移动**和**代码重构**等4个部分

### 1. 总体设计

项目整体分为8个模块，看图。（づ￣3￣）づ╭❤～

<div align=center>
<img src="/medias/BlogPictures/1. AimYolo/pictures/1. 总体思路.png" width = 100%>
</div>

- **参数获取**：继承自YOLO，个性化参数，如指定模型对目标的检测部位（头部、身体等）
- **屏幕截取**：CPU或GPU截图，截图时间尽可能短，可指定屏幕截图区域
- **预处理**：截图数据转为tensor数据类型，与YOLO接口一致
- **模型推理**：图像数据交付核心模型，进行检测
- **坐标计算**：根据模型的推理数据，计算目标位置的坐标
- **鼠标移动**：将鼠标移动至对应坐标，如头部、身体及其中心位置
- **专用/通用数据集**：如CSGO数据集（标注CT，CT_Head，T，T_Head），传统人体识别数据集等
- **模型训练**：由于待检测目标一般较少，且对速度要求较高，选取s版本的模型
- **整体**：整体采用单进程，在While循环中持续检测，并设置信号用于结束进程，后续完善为根据指定键位，更改AI姿态（大概会吧T_T）
---

### 2. 屏幕实时捕获

这里并非真正的实时，而是从屏幕截取到模型推理，再到鼠标移动，整个过程的用时很少，宏观感受为实时。废话少说，看图（づ￣3￣）づ╭❤～。

<div align=center>
<img src="/medias/BlogPictures/1. AimYolo/pictures/2. 屏幕实时获取.png" width = 70%>
</div>

- **CPU截屏**：采用mss库进行截屏；也可以采用d3dshot库进行GPU截屏，但其中会涉及numpy和tensor数据的运算速度、显卡的设置问题，不推荐
- **实时截图与展示**：将截取的画面实时展示，便于测试与直观感受，根据参数配置作为可选项

以只因哥打篮球的视频为例，这个部分的效果应该是这样的。（づ￣3￣）づ╭❤～

<!-- <div align=center>
<iframe height=360 width=640 frameborder=0 src="/medias/video/cxk.mp4">
</div> 这样写显示有问题-->

<div align=center>
<video width="640" height="360" controls>
<source src="/medias/video/cxk.mp4">
</video>
</div>

---

### 3. 鼠标定位与移动
关于鼠标的定位与移动，需要高清几个问题，看图（づ￣3￣）づ╭❤～。

<div align=center>
<img src="/medias/BlogPictures/1. AimYolo/pictures/3. direct-input.png" width = 100%>
</div>

- **鼠标定位**
  - 即让鼠标搞清自己的定位
  - step1. 获取鼠标当前位置的坐标/像素
  - step2. 结合YOLO，根据检测框计算坐标
  - step3. 计算欧式距离，移动鼠标
  - **特别注意**（部分问题会在**核心代码**部分解释）
    - 鼠标移动时的绝对距离与相对距离（一般为相对距离，故计算了欧氏距离）
    - 物理距离与分辨率不一致的问题（开启DPI感知）
    - 一些游戏需要关闭原始输入
  
- **鼠标移动**
  - A. Windows程序鼠标移动机制
    - 如图，传统机制已经满足不了当下大型游戏的需求
  - B. DirectX游戏开发
    - 微软开发的多媒体编程接口，具有更快的响应速度
    - 导致PyAutoGui、Pynput等采用Virtual Key Codes的键鼠控制库失效
  - C. 绕过限制
    - 更先进的Scan Codes和Win32 Sendinput方法，实现游戏内的键鼠快速控制
    - 如图，采用了**Ctypes库**
    - 在一些老游戏，如CS1.6中，则不用考虑这些
---

### 4. 代码重构

重构了YOLOv5中的detect.py：保留核心模型，更改原有逻辑，添加自定义功能。看图。（づ￣3￣）づ╭❤～

<div align=center>
<img src="/medias/BlogPictures/1. AimYolo/pictures/4. 代码重构.png" width = 100%>
</div>

更多细节留给**核心代码**部分。

---

## 三、模型开发
本节分为**模型结构**、**数据集**和**模型训练**等3个部分。

### 1. 模型结构
采取的模型结构，看图。具体介绍可参考[江佬的博客](https://blog.csdn.net/nan355655600/article/details/107852353)，该图也源此。（づ￣3￣）づ╭❤～

<div align=center>
<img src="/medias/BlogPictures/1. AimYolo/pictures/5. yolov5s.PNG" width = 100%>
</div>

需要说明的是：
  - **代码版本**：YOLOv5-2.0
  - **选取模型**：Pretrained YOLOv5s，后续训练时，以此为依托，进行不太严格的fine-tune
  - **任务类型**：目标检测
---

### 2. 数据集
本小节对使用的数据集进行介绍。
1. **CSGO专用数据集**  
   [数据集项目地址](https://github.com/qcjxs-hn/yolov5-csgo)。时间有限，并没有进行数据集的构建，当时，这个项目解决了博主的燃眉之急。值得说明的是，该项目的创作者也附加了一些单纯的检测代码。
2. **检测类别数：4**  
   **ct, t, t_head, ct_head**，即反恐精英，恐怖分子，恐怖分子头部，反恐精英头部。这有利于模型进行目标身体或头部的检测与锁定。
3. **数据集划分**  
   图片总数：800张。博主进行主观划分->训练集：700张，验证集：100张。
4. **数据分布特征**  
   <div align=center>
    <img src="/medias/BlogPictures/1. AimYolo/pictures/6. labels.png" width = 100%>
    </div>
5. **数据示例**  
   img_250.jpg的**原始图像信息**、**标签信息**，以及**一些图片的groundtruth**如下。（づ￣3￣）づ╭❤～
    <div align=center>
    <img src="/medias/BlogPictures/1. AimYolo/pictures/7. img_250.jpg" width = 100%>
    </div>
    <div align=center>
    <img src="/medias/BlogPictures/1. AimYolo/pictures/8. img_205_label.PNG" width = 100%>
    </div>
    <div align=center>
    <img src="/medias/BlogPictures/1. AimYolo/pictures/9. test_batch0_gt.jpg" width = 100%>
    </div>
---

### 3. 模型训练



## 四、代码说明


## 五、项目部署与展示


## 六、总结


## 参考

