#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

import numpy as np

# 1、验证np.max的作用
u = np.array([[1, 2, 3], [4, 5, 6]])
print(u)
v = np.max(u, 0)
print(v)
print('1Finished\n')

# 2、验证np.maximum的作用
i = np.array([[1, 3, 0], [5, 4, 0]])
j = np.array([[1, 2, -2], [4, 5, -1]])
k = np.maximum(i, j)
l=np.maximum(i,1)
print(k)
print("")
print(l)
print('2Finished\n')

# 3、验证+和，在print中的区别
a = '你'
b = '谁'
print(a + b)
print('3Finished\n')
