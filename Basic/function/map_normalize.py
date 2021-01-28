#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: map_normalize.py
# author: twh
# time: 2020/9/23 9:12


def normalize(name):
    name = name[0].upper() + name[1:].lower()
    return name


l_0 = ['admin', 'JACK', 'banB']
l_1 = list(map(normalize, l_0))
print(l_1)
