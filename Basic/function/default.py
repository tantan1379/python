#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: default.py
# author: twh
# time: 2020/9/22 17:34


def powerofn(a, default = 2):   # 可以指定任意次方，但如果不指定，则默认为平方
    result = 1
    for i in range(default):
        result *= a
    return result


print(powerofn(5, 3))

