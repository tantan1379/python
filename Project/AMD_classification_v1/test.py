#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2021/3/11 20:24
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import glob
import numpy as np

# root_path = r"F:/Lab/AMD_CL/origin/2/01-A-0036-V1-OCT"  # 数据集的根目录
# for file in os.listdir(root_path):
#     images = []
#     images += glob.glob(os.path.join(root_path, "*.jpg"))  # 将所有图片的路径写入到images列表中
#     for image in images:
#         img = Image.open(image)
#         cropped = img.crop((500, 0, 1008, 435))
#         cropped.save("F:/Lab/AMD_CL/preprocessed/temp/"+image.split(os.sep)[-1])
#     # print(len(images))

# pic_path=r"F:\Lab\AMD_CL\preprocessed\1\1陈000.jpg"
# img = Image.open(pic_path).convert('RGB')
# img = Image.open(pic_path)
# plt.imshow(img)
# plt.show()

# sensi = np.array([-1] * 3)
# print(sensi)


