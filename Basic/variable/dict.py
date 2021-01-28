#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: dict.py
# author: twh
# time: 2020/9/20 20:11

# 一、创建
# 创建空字典
a = {}
print(a)

# 创建一个带数字的字典
c = {'tan': 1, 'b': 'c', 'd': [1, 2]}
print(c)

# 多行代码创建字典
d = {
    'name': 'wen',
    'age': 19,
    'gender': '男'
}
print(d)

# 用dict函数创建字典:
b = dict(name = 'tan', age = 18, gender = '男')  # 默认关键字为一个字符串
print(b)
e0 = [1, 2, 3]
e1 = [5, 6, 7]
print(list(zip(e0, e1)))
e = dict(zip(e0, e1))  # 映射函数方式创建字典
print(e)
f = dict([(1, 3), ('a', 'b')])  # 可迭代对象方式创建字典
print(f)

# 二、查找
# 用get函数获取字典的value
g = dict(a = 1, b = 2, c = 3)
print(g)
print(g.get('a'))
print(g.get('d', 'not found'))   # 找不到键时，返回第二个填入的值

# 三、修改
# setdefault方法，第一个位置填key，第二个位置填value。如果key已存在，不管给何value都不改变字典，且返回原字典中key所对应的value；
# 如果key不存在，则添加该key-value，且返回添加的value
h0 = dict(a = 1, b = 2, c = 3)
h0.setdefault('d', 4)  # 键不存在，则添加
print(h0)
h0.setdefault('c', 4)  # 键已存在，不改变
print(h0)

# 用del删除字典中的内容
h1 = dict(a = 1, b = 2, c = 3)
del h1['a']  # 不存在则报错
print(h1)

# 用popitem删除字典中的内容（随机）
h2 = dict(a = 1, b = 2, c = 3, d = 4)
h3 = h2.popitem()
print(h2)
print(h3)   # popitem将返回一个元组

# 用pop删除字典中的内容
h4 = dict(a = 1, b = 2, c = 3)
print(h4.pop('d', 10))  # 不存在则返回设定的default,且不改变字典的内容
print(h4)
print(h4.pop('a'))  # 存在则删除指定键值对，且返回被删除的value
print(h4)

# 四、遍历
i = dict(a = 1, b = 2, c = 3)
print(type(i.keys()))
for i0 in i.keys():  # keys()方法将会返回字典所有key的一个序列
    print(i0, i[i0])

print(type(i.values()))
for i1 in i.values():  # values()方法将会返回字典所有value的一个序列
    print(i1)

print(type(i.items()))
print(i.items())  # items()方法将会返回包含字典双值子序列的一个序列
for i2, i3 in i.items():
    print(i2, i3)
for i4 in i.items():  # i4为双值子序列
    print(i4)

# 五、其他
j = dict(a = 1, b = 2, c = 3)
print(len(j))  # len()函数获取键值对的个数
print('b' in j)  # 检查键是否在字典中
