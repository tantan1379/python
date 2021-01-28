#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: bubble_sort.py
# author: twh
# time: 2020/9/22 17:07

def sort(a_list):
    for i in range(len(a_list) - 1):  # 如果列表有n个元素，则最多需要进行n-1次冒泡排序
        flag = 0  # 预设一个标志，用于检测序列是否被交换
        for j in range(len(a_list) - i - 1):
            if a_list[j] > a_list[j + 1]:
                a_list[j], a_list[j + 1] = a_list[j + 1], a_list[j]
                flag += 1  # 如果进行了交换，就将flag变化
        if flag == 0:  # 如果flag没被取反，说明每次两两比较都符合从小到大的要求，排序已经完成
            return a_list  # 返回调整好的列表，并退出该循环以及该函数
    return a_list


my_list = [5, 4, 3, 2, 1]
print(sort(my_list))
