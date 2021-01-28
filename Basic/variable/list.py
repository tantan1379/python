#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

# 列表的方法（方法用”对象.方法“的形式调用）
j = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(j.index(2))
print(j.count(1))
j.append('10')           # 只能加到列表最后
print(j)
j.insert(1, 1)           # 第一个数填要加入的索引位置，第二个填加入的内容
print(j)
j.extend([11, 12, 13])   # 只能填一个数字
print(j)
print(j.pop(13))         # 删除索引为13的数字且可以返回该数字（用于删除可索引的值）
print(j)
j.remove(1)              # 只会删第一个要求的数字（用于删除知道内容的值）
print(j)
j.reverse()              # 不用输参数
print(j)
j.remove('10')
print(j)
j.sort()                 # 只能用于全为数字或全为字符的列表
print(j)


# 用for循环遍历列表
my_list = ['tan', 'wen', 'hao']
for name in my_list:
    name *= 2
    print(name)