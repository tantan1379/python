#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: enumerate.py
# author: twh
# time: 2020/10/28 16:08



import numpy as np

X = np.array([3, 1, 3, 9, 5, 6, 7, 1, 4, 2, 4, 8, 1, 3, 4, 7])
T = np.zeros((X.size, 10))

for idx, row in enumerate(T):
    print(idx,row)
    row[X[idx]] = 1

print(T)