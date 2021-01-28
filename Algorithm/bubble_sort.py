#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author:twh time:2020/9/18


# 实现冒泡排序，输入无序列表，返回有序列表
def bubble_sort(arr):
    for i in range(len(arr) - 1):
        flag = True
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                flag = False
        if flag:  # 如果flag依然为True则说明未进行交换，说明序列已经排好，可以跳出循环
            return arr
    return arr


a = [4, 5, 3, 7, 4, 6, 1, 4, 3, 4]
bubble_sort(a)
print(a)
