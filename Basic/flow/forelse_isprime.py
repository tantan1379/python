#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

# 用for..else判断是否为质数
for i in range(10, 20):
    for j in range(2, i):
        if i % j == 0:
            k = i / j
            print('%d不是质数,且%d等于%d*%d' % (i, i, k, j))
            break
    else:
        print('%d是质数' % i)
