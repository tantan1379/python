#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: student.py
# author: twh
# time: 2020/9/24 21:25


class Student:
    _age = ''

    def __init__(self, name, music, age):
        self.name = name
        self.music = music
        self.set_age(age)

    def get_age(self):
        return self._age

    def set_age(self, age):
        if 0 < age < 120:
            self._age = age
        else:
            self._age = 18


class Teacher(Student):
    award = ''

    def __init__(self, name, music, age, award):
        super().__init__(name, music, age)
        self.award = award


if __name__ == '__main__':
    Jian = Student('邵子健', '江南', 22)
    print(Jian.get_age())
    Jian.set_age(23)
    print(Jian.get_age())
    Yu = Teacher('郁连国', '曹操', 23, '一等奖')
    print(Yu.award)
    print(Yu.get_age())
