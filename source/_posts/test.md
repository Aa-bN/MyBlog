---
title: test
date: 2022-09-01 21:12:44
author: 好想摸鱼
img:
top: true
cover: false
coverImg:
password:
toc: true
mathjax: true
summary: 这是一篇测试用文章,，本段为摘要
categories: 测试
tags:
    - test
    - mass
keywords:
reprintPolicy: cc_by
---
## 1. 搭建方式
> Hexo + Github + Netlify  
> 
> 未购买域名 
> 
> 未使用加速服务
## 2. 代码
```python
class Resnet34FCN(nn.Module):
    def __init__(self):
        super(Resnet34FCN, self).__init__()
        # 缩小8倍, 3*3, 128
        self.stage1 = nn.Sequential(*list(pretrained_resnet34.children())[:-4])
        # 缩小16倍, 3*3, 256
        self.stage2 = list(pretrained_resnet34.children())[-4]
        # 缩小32倍, 3*3, 512
        self.stage3 = list(pretrained_resnet34.children())[-3]

        # 三个1*1卷积操作，信息融合
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        # 将特征图尺寸放大八倍
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        # 2x，为了区分写成4
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        # 2x
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 224 / 8 = 28, /8

        x = self.stage2(x)
        s2 = x  # 28 / 2 = 14, /16

        x = self.stage3(x)
        s3 = x  # 14 / 2 = 7, /32

        s3 = self.scores1(s3)       # 7
        s3 = self.upsample_2x(s3)   # 14
        s2 = self.scores2(s2)       # 14
        s2 = s2 + s3                # 14*14

        s1 = self.scores3(s1)       # 28
        s2 = self.upsample_4x(s2)   # 28
        s = s1 + s2                 # 28*28

        s = self.upsample_8x(s)     # 224*224

        return s
```

## 3. 数学公式
$y = f(x)$  
$$\sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6}$$