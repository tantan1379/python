#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: variable_object.py
# author: twh
# time: 2020/9/20 16:21

# # 修改一个列表
# a = [1, 2, 3]
# print("a", a, id(a))
# print("")
#
# b = a
# b[0] = 10
# print("a", a, id(a))
# print("b", b, id(b))
# print("")
#
# a[0] = 3
# print("a", a, id(a))
# print("b", b, id(b))
# print("")
#
# b = [10, 2, 3]
# print("b", b, id(b))

# 修改一个int数据
c = 1
print("c", c, id(c))
b = c
print("b", b, id(b))
c = 2                           # 属于重新赋值，会改变id
print("c", c, id(c))
print("b", b, id(b))            # 不属于可变对象，因此将c赋给b之后，两者不会同步改变，但id相等

# # 修改一个元组
# a = 1, 2, 3, 4, 5
# b = a
#
# print("a", a, id(a))
# print("b", b, id(b))
