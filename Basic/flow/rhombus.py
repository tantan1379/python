#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17


# 原
def print_rhombus(length):
    for i in range(length):
        for j in range(length - i):
            print(" ", end = "")
        for k in range(2 * i - 1):
            print("*", end = "")
        print("")
    for i in range(length):
        for j in range(i):
            print(" ", end = "")
        for k in range(2*length-1-2*i):
            print("*", end = "")
        print("")


# 优化
def print_rhombus_op(length):
    for i in range(length):
        print(" " * (length - i), end = "")
        print("*" * (2 * i - 1), end = "")
        print("")
    for i in range(length):
        print(" " * i, end = "")
        print("*" * (2 * length - 1 - 2 * i), end = "")
        print("")


print_rhombus_op(5)
