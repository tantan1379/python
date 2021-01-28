#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: select_sort.py
# author: twh
# time: 2020/10/23 22:07


#  找出列表中最小数字的索引
def findsmallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index


#  实现选择排序
def select_sort(arr_mine):
    arr_temp=list(arr_mine)  # "[:]"赋值而非引用，使排序不影响原数组
    arr = []
    for i in range(len(arr_temp)):  # 每次都将需排序数组最小的元素移除，移除后就会出现新的最小值，循环len(arr)次，并依次添加到arr数组中
        smallest_index = findsmallest(arr_temp)
        arr.append(arr_temp.pop(smallest_index))
    return arr


#  实现选择排序（输入数组被改变）
def select_sort_1(arr_mine_1):
    for i in range(len(arr_mine_1) - 1):
        for j in range(len(arr_mine_1) - i - 1):
            if arr_mine_1[j] > arr_mine_1[j + 1]:
                arr_mine_1[j], arr_mine_1[j + 1] = arr_mine_1[j + 1], arr_mine_1[j]
    return arr_mine_1


if __name__ == "__main__":
    myarr = [5, 4, 2, 3, 1]
    arr_sort = select_sort(myarr)
    print(arr_sort)
    # myarr_1 = [5, 4, 2, 3, 1]
    arr_sort_1 = select_sort_1(myarr)
    print(arr_sort_1)
