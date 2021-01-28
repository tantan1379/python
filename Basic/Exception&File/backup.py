#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: backup.py
# author: twh
# time: 2020/11/8 18:06

def Backup():
    old_file = input('please input the origin file name:')
    namelist = old_file.split('.')
    new_file = namelist[0] + '_backup.' + namelist[1]
    try:
        with open(old_file, 'rb',encoding='utf-8') as old_f, open(new_file, 'wb') as new_f:
            while True:
                content = old_f.read(1024)
                new_f.write(content)
                if len(content) < 1024:
                    break
    except Exception as msg:
        print(msg)
        print(msg.__class__)

Backup()