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
    ---

本项目理论上可以用于CSGO、APEX、PUBG等多种游戏，实际以CSGO为例进行了模型的开发与使用。相对于传统外挂，AI外挂多基于目标检测算法，不会修改游戏的本地内存，也不会上传恶意数据。本文后续有6个部分：**开发思路**、**模型训练**、**代码讲解**、**项目部署与展示**、**总结**与**参考**。

---

## 二、开发思路
本节分为**总体设计**、**屏幕实时捕获**、**鼠标定位与移动**和**代码重构**等4个部分

### 1. 总体设计
<div align=center>
<img src="/medias/BlogPictures/1. AimYolo/pictures/1. 总体思路.png" width = 100%>
</div>

### 2. 屏幕实时捕获

### 3. 鼠标定位与移动

### 4. 代码重构


