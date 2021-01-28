#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: file.py
# author: twh
# time: 2020/9/27 14:02
with open(file='1.txt',mode='w') as f:
    f.write('你好')
with open(file='1.txt',mode='a') as f:
    f.write('\n菜鸟')

with open(file='1.txt',mode='rb') as f:
    f1=f.read()
    print(f1)
    print(f1.decode('gbk'))


