#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py.py
# author: twh
# time: 2021/1/22 15:22
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import zipfile
print(os.listdir(r"F:\Database"))
with zipfile.ZipFile(r"F\Database\train.zip") as z:
    z.extractall(".")

