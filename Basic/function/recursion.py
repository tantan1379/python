#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: recursion.py
# author: twh
# time: 2020/9/22 20:09


def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


print(factorial(10))


def fibonacci(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


for i in range(1, 10):
    print('第%d个数为:' % i, fibonacci(i))


def totalmoney(money, year):
    if year == 1:
        return money * 1.02
    else:
        return totalmoney(money, year - 1) * 1.02


print(totalmoney(1000, 2))

