#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/22 16:46

def fun(a, **kwargs):
    print('a is', a)
    print('b is', kwargs['b'])
    print('c is', kwargs['c'])


fun(1,**{'b':1,'c':3})