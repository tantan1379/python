#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: dog.py
# author: twh
# time: 2020/9/24 21:12


class Dog:
    _color = ''

    def __init__(self, color, food, name, age):
        self._color = color
        self.food = food
        self.name = name
        self.age = age

    def eat(self):
        print('我的名字是{}，我今年{}岁了，我最爱吃的是{}'.format(self.name, self.age, self.food))

    def run(self):
        print('我的名字是{}，我今年{}岁了，我跑起来{}的毛色很好看'.format(self.name, self.age, self._color))

    def get_color(self):
        return self._color

    def set_color(self, color):
        self._color = color


if __name__ == '__main__':
    dog1 = Dog('black', 'chicken', 'wangcai', 5)
    print(dog1.get_color())
    dog1.set_color('red')
    print(dog1.get_color())
    dog1.eat()
    dog1.run()
