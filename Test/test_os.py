#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test_os.py
# author: twh
# time: 2020/11/18 13:14
import os
import glob

# namelist = {}
# for index, name in enumerate(sorted(os.listdir(os.path.pardir))):
# namelist[index] = name
# namelist[name] = len(namelist.keys())
root = 'Basic'
labels = {}
files = []
for name in os.listdir(os.path.join(os.path.pardir, root)):
    if not os.path.isdir(os.path.join(os.path.pardir, root, name)):
        continue
    labels[name] = len(labels.keys())
# print(labels)
for name in labels.keys():
    # print(name)
    files += glob.glob(os.path.join(os.path.pardir, root, name,'*.py'))
    files += glob.glob(os.path.join(os.path.pardir, root, name,'*.txt'))

print(files)
