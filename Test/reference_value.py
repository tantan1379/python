#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: reference_value.py
# author: twh
# time: 2020/12/8 20:01

def bad_append(new_item, a_list=[]):
    a_list.append(new_item)
    return a_list


def good_append(new_item, a_list=None):
    if a_list is None:
        a_list = []
    a_list.append(new_item)
    return a_list


print(bad_append(1))  # output:[1]
print(bad_append(1))  # output:[1,1]
print(good_append(1))  # output:[1]
print(good_append(1))  # output:[1]
