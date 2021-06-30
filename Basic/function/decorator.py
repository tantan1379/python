#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: decorator.py
# author: twh
# time: 2020/9/23 10:12

import time


def decorator(fun):
    def wrap():
        tic = time.time()
        print('started')
        fun()
        toc = time.time()
        print('It consumes', str((toc - tic) * 1000), 'ms')

    return wrap


@decorator
def needed_fun():
    print('i am needed')


needed_fun()
