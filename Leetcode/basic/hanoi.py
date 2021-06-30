#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: hanoi.py
# author: twh
# time: 2021/3/7 19:44
def hanoi(n, a, b, c):
    if n > 0:
        hanoi(n - 1, a, c, b)
        print("moving from %s to %s" % (a, c))
        hanoi(n - 1, b, a, c)


hanoi(4, 'A', 'B', 'C')
