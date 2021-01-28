#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: inherit.py
# author: twh
# time: 2020/9/26 14:39


class Animals:
    def __init__(self, color):
        self._color = color


class Dogs(Animals):
    def __init__(self, color, age):
        super().__init__(color)
        self._age = age

    @staticmethod
    def run():
        print('跑步')

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        self._age = age

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color


if __name__ == '__main__':
    d = Dogs('black', 10)
    d.run()
    print(d.color)
    d.color = 'red'
    print(d.color)
