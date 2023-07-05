# Chinese_Handwriting_Recognition
HEBUT大四上学期校内实习项目 基于OpenCV和CNN的汉字手写识别系统

![svg](https://forthebadge.com/images/badges/made-with-python.svg)
![svg](https://forthebadge.com/images/badges/made-with-javascript.svg)
[![svg](https://github.com/WangJerry1229/WangJerry1229/raw/main/badge.svg)](https://wangjiayi.cool)

## 项目介绍
1. Model 
   - HWDB1.0数据集处理脚本 (hwdb.py)

     数据集下载地址: http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

   - MobileNet V2网络结构
   - 添加了一个基于LeNet5的基础版本 (\LeNet5)
   - 模型训练测试 (train.py test.py)
   - 基于水平垂直投影分割的位置检测 (projection_detect.py）
3. Server
   - 基于Flask的服务端 (server.py)
4. Web
   - 用户端网页展示 (index.html multiple.html)
   - 基于Canvas的网页手写输入

## 在线演示 (不确保正常运行)
    https://wangjiayi.cool/ocr/

## 未来计划（还不知道什么时候再研究）

- [ ] 扩大训练集，支持更多常用汉字
- [ ] 训练Detect模型，替换掉现有的分割算法
- [ ] 优化网络结构
