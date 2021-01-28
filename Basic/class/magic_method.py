#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: magic_method.py
# author: twh
# time: 2020/11/17 20:38

# class Test(object):
#     def __init__(self,dataset, name):
#         self.name = name
#         self.dataset = dataset
#
#     def __str__(self):  # 允许直接在类中填入字符串，并返回一个值
#         return 'string of Test is {}'.format(self.name)
#     __repr__ = __str__  # 用于程序员调试，在控制台直接输入类也能显示返回值
#
#     def __getitem__(self,index):  # 得到数据库指定索引的batch
#         return self.dataset[index]
#
#     def __len__(self):  # 得到数据库样本数
#         return len(self.dataset)
#
#     def __iter__(self): # 返回一个可迭代对象
#         return self
#
#
# Test('abc')
from torch.utils.data import DataLoader


class diy():

    def __len__(self):  # 必须要有！
        print('len函数被调用了')
        return 4

    def __getitem__(self, index):  # 必须要有！
        print('getitem函数被调用一次，这次的index是{}'.format(index))
        return '并肩于雪山之巅！！' * (index + 1)


b = diy()

b_batch = DataLoader(b, batch_size=1, shuffle=False) # 先调用__len__获得数据库的长度

for batch_idx, output_sentence in enumerate(b_batch):  # 每次循环依次获得2个__getitem__的返回值
    print('这是第{}个batch'.format(batch_idx + 1))
    for i in range(len(output_sentence)): # 这里的len不是类中的魔术方法，而是output_sentence本身的长度，长度等于batch_size
        print(output_sentence[i])
