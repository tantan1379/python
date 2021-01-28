#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: draw_tangent_line.py
# author: twh
# time: 2020/10/28 16:38
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    y=0.01 * x ** 2 + 0.1 * x
    return y


def compute_diff(func, x0):
    h = 1e-5
    return (func(x0 + h) - func(x0 - h)) / (2 * h)


def tangentline(func, x0):
    k = compute_diff(func, x0)
    b = func(x0) - k * x0
    y = lambda x: x * k + b        # 建立映射关系 y(x)=kx+b
    return y


def main():
    x = np.arange(0.0, 20.0, 0.1)
    y=function(x)
    yf=tangentline(function,5)
    y_tangent=yf(x)
    plt.plot(x,y)
    plt.plot(x,y_tangent)
    plt.show()


if __name__ == "__main__":
    main()
