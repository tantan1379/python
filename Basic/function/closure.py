#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: closure.py
# author: twh
# time: 2020/9/23 9:30


def make_average():
    nums = []

    def averager(n):
        nums.append(n)
        return sum(nums) / len(nums)

    return averager


avg = make_average()
print(avg(10))
print(avg(20))
print(avg(30))


def test():
    a = 0
    b = 0

    def add(a, b):
        print(a + b)

    return add


g = test()
g(10, 11)
