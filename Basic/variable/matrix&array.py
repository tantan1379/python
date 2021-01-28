#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

import numpy as np

num = [2, 2, 3]
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.mat([[1, 2, 3], [4, 5, 6]])
c = np.mat(np.zeros((3, 3)))
d = np.zeros(num)
e = np.random.rand(3, 3)
f = np.mat(np.ones((2, 3)))
g=a[0,:].max()

# others(a)
# others(b)
# others(c)
# others(d)
# others(e)
# others(f)
print(g)
