#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17


import time
import calendar

a = time.asctime(time.localtime())
c = time.localtime()
print(a)
print("")
print(c)
print("")
b = calendar.month(2020,9)
print(b)