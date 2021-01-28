#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: id.py
# author: twh
# time: 2020/12/8 21:13

def bar(args):
    print(id(args))  # output:4324106952
    args.append(1)


b = []
print(b)  # output:[]
print(id(b))  # output:4324106952
bar(b)
print(b)  # output:[1]
print(id(b))  # output:4324106952
