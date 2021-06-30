#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17

# import numpy as np

fruit = ['grape', 'apple', 'orange']

# 1、用字符串进行循环
for i in fruit:
    print('当前水果:', i)

for j in 'python':
    print('当前字母:', j)

print('1Finished\n')

# 2、用索引进行循环
for index in range(len(fruit)):
    print('该水果为:', fruit[index])

print('2Finished\n')

# 3、用range进行累加

for n in range(5):
    print(n)
    n += 1

print('3Finished\n')

# 4、测试+=
a = 0
for d in range(0, 5):
    a += d
    print(a)

print('4Finished\n')

# 5、斐波那契数列
f1 = 1
f2 = 1
for i in range(1, 22):
    print('%12d %12d' % (f1, f2), end = "")
    if i % 3 == 0:
        print("")
    f1 = f1 + f2
    f2 = f1 + f2
print('5Finished\n')

# 6、格式化操作符辅助符(dict）
students = [{"name": "Susan", "age": 20}, {"name": "YYM", "age": 21}]

for student in students:
    print("%(name)s is %(age)d years omd" % student)

print('6Finished\n')

# 7、格式化操作符辅助符(其他）
a = 10
b = 10.110000111
print("%+d \n %.*f" % (a, 20, b))
print('7Finished\n')

# 8、用列表输出斐波那契数列
u = []
for i in range(21):
    u.append([])

u[0] = 1
u[1] = 1

for j in range(2, 21):
    u[j] = u[j - 1] + u[j - 2]
for m in range(len(u)):
    print("%10d" % u[m], end = "")
    if (m + 1) % 3 == 0:
        print("")
print('8Finished\n')
