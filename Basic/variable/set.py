#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: set.py
# author: twh
# time: 2020/9/21 16:33

# 一、集合的创建
# 创建一个空集合
a = set()  # 只能用set()，不能用{}
print(a)

a0 = {1, 2, 3, 4, 5, 6, 7}
print(type(a0))
print(a0)
a1 = set([1, 2, 3, 4, 5])  # 实际上此时是将序列转换为集合
print(a1)
a2 = set({'a': 1, 'b': 2})  # 将字典转换为集合，只包含字典的keys
print(a2)

# 二、集合的函数
b = {1, 2, 3, 4, 5}
b.add(6)
print(b)
b.remove(5)
print(b)
b.update({1, 2, 7, 8})
print(b)

