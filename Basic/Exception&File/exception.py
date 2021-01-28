#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: exception.py
# author: twh
# time: 2020/9/27 9:53


try:
    a = [1, 2, 3, 4]
    print(a[5])
except NameError:
    print('Name')
except ZeroDivisionError:
    print('ZeroDivision')
except IndexError:
    print('Index')
except Exception as e:
    print('unknown',e,type(e))
finally:
    print('next')