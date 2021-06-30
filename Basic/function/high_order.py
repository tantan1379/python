#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: high_order.py
# author: twh
# time: 2020/9/22 20:53

# 简单加法
def getsum(x, y, fn):
    return fn(x) + fn(y)


print(getsum(1.1, 2.1, round))


# 寻找序列中的偶数
def findeven(list1):
    l1 = []
    for i in list1:
        if i % 2 == 0:
            l1.append(i)
    return l1


def printeven(list1, fn):
    l2 = fn(l0)
    print(l2)


l0 = [1, 2, 3, 4, 5, 6, 7, 8]
printeven(l0, findeven)


# 计算两个数阶乘的和

def get_sum(x, y, fn):
    return fn(x) + fn(y)


def get_factorial(num):
    result = 1
    for i in range(1, num + 1):
        result *= i
    return result


print(get_sum(2, 5, get_factorial))
