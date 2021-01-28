#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: name_passage.py
# author: twh
# time: 2020/9/24 20:55


class Dor606:
    def __init__(self, name, gender, password):
        self.name = name
        self.gender = gender
        self.password = password

    def outprint(self):
        print('用户名为{},密码为{} '.format(self.name, self.password),end="")

    def print_gender(self):
        print('性别为{}'.format(self.gender))


a = Dor606('谈文昊', '男', '123123')  # 实例化对象
a.outprint()
a.print_gender()
b = Dor606('邵子健', '女', '521521')  # 实例化对象
b.outprint()
b.print_gender()