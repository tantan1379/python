#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author:twh time:2020/9/17


# 实现二分查找功能，输入列表（要求列表元素是有序的）和需要查找的数字，返回该数字在列表的索引
def binary_search(seq, item):
    low = 0
    high = len(seq) - 1
    while low <= high:
        mid = (low + high) // 2
        guess = seq[mid]
        if guess == item:
            return mid
        elif guess < item:
            low = mid + 1
        else:
            high = mid - 1
    return None


seq_mine = [1, 3, 4, 6, 7, 9]
tag = binary_search(seq_mine, 3)
print(tag)
