#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

# 1、元组的解包
a = 1, 2, 3, 4, 5, 6            # 不加括号且有元素时时默认为元组
c,*b = a  # 解包
print(b)
