#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: os_scandir.py
# author: twh
# time: 2020/11/8 20:42
import os

path='f:/git'

dir=os.scandir(path)
for name in dir:
    print(name.name)
