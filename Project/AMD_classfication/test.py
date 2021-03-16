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

root_path = r"F:/Lab/AMD_CL/origin/2/01-A-0036-V1-OCT"  # 数据集的根目录
for file in os.listdir(root_path):
    images = []
    images += glob.glob(os.path.join(root_path, "*.jpg"))  # 将所有图片的路径写入到images列表中
    for image in images:
        img = Image.open(image)
        cropped = img.crop((500, 0, 1008, 435))
        cropped.save("F:/Lab/AMD_CL/preprocessed/temp/"+image.split(os.sep)[-1])
    # print(len(images))
