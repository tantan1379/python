#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17


# a = 4
# b = 2
# others('a>b') if (a > b) else others('b>a')
# max_a_b = a if (a > b) else b
# others(max_a_b)

import numpy as np
import random

a=random.uniform(-10,1000)
b=np.random.randint(-1000,100)

print("a=%.10f,b=%d" %(a,b))