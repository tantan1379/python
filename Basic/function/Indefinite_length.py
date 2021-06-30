#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: Indefinite_length.py
# author: twh
# time: 2020/9/22 18:41


# def getsum(*num):
#     count = 0
#     for i in num:
#         count += i
#     return count
#
#
# def getaver(*num_0):
#     count_0 = 0
#     for j in num_0:
#         count_0 += j
#     return count_0 / len(num_0)
#
#
# def newprint(*cont):    # 无缝输出
#     for i in cont:
#         print(i, end = "")
#
#
# print(getsum(10, 11, 12))
# print(getaver(10, 11, 12))
# newprint('a', 'b')


def arg_print(farg, *args):  # 变量前加*相当于解包(unpack)
    print("farg is", farg)
    for arg in args:
        print("args contain", arg)


arg_print(1, 2, 3, 4, 5)


def get_sum(*args):
    return sum(args)


def get_other_sum(a, *args):
    result = get_sum(*args)
    return result


print(get_other_sum(1, 4, 5))
