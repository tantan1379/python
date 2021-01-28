#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

for i in range(1, 10):
    for j in range(1, i + 1):
        print("%d*%d=%2d" % (i, j, i * j), end = " ")
    print("")
