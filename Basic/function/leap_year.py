#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: leap_year.py
# author: twh
# time: 2020/9/22 16:50

def find_leap_year(x, y):
    if x >= y:
        print('输入年份有误')
    else:
        for i in range(x, y + 1):
            if (i % 4 == 0 and i % 100 != 0) or i % 400 == 0:
                print(i, end = " ")


find_leap_year(400, 1200)
