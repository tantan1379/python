#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: 155_Min_Stack.py
# author: twh
# time: 2020/12/20 17:10

"""
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
    push(x) -- Push element x onto stack.
    pop() -- Removes the element on top of the stack.
    top() -- Get the top element.
    getMin() -- Retrieve the minimum element in the stack.

设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
    push(x) —— 将元素 x 推入栈中。
    pop() —— 删除栈顶的元素。
    top() —— 获取栈顶元素。
    getMin() —— 检索栈中的最小元素。
"""


class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.q, self.temp = [], []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.q.append(x)
        if not self.temp:
            self.temp.append(x)
        else:
            if x > self.temp[-1]:
                self.temp.append(self.temp[-1])
            else:
                self.temp.append(x)
        print("此时：\nq为：",self.q)
        print("temp为：",self.temp)

    def pop(self):
        """
        :rtype: None
        """
        self.temp.pop()
        self.q.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.q[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.temp[-1]


# Your MinStack object will be instantiated and called as such:
obj = MinStack()
obj.push(-2)
obj.push(0)
obj.push(-3)
print(obj.getMin())
obj.pop()
print(obj.top())
print(obj.getMin())
