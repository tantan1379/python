#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

num = 0
for num in range(1000):
    ge = num % 10
    shi = num % 100 // 10
    bai = num // 100
    if ge ** 3 + shi ** 3 + bai ** 3 == num:
        print('%4d是水仙花数' % num)
