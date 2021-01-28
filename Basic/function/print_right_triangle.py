#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: print_right_triangle.py
# author: twh
# time: 2020/9/22 17:03


def print_triangle(n):
    for i in range(1, n + 1):
        for j in range(i):
            print("*", end = "")
        print("")


print_triangle(10)
