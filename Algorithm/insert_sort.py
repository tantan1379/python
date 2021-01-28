#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: insert_sort.py
# author: twh
# time: 2020/10/26 16:21

import numpy as np
import time

def insert_sort(arr_input):
    arr = arr_input[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while (j >= 0) and (arr[j] > key):
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def insertsort(arr_input):
    arr = arr_input[:]
    for i in range(1, len(arr)):
        j = i
        while j > 0:
            if arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                j -= 1
            else:
                break
    return arr


def main():
    arr = np.random.randn(10000)
    tic=time.time()
    insert_sort(arr)
    toc=time.time()
    print(toc-tic)
    tic=time.time()
    insertsort(arr)
    toc=time.time()
    print(toc-tic)


if __name__ == "__main__":
    main()
