"""
Implement a last in first out (LIFO) stack using only two queues. The implemented stack should support all the functions
of a normal queue (push, top, pop, and empty).

使用队列实现栈的下列操作：

    push(x) -- 元素 x 入栈
    pop() -- 移除栈顶元素
    top() -- 获取栈顶元素
    empty() -- 返回栈是否为空

注意:

    你只能使用队列的基本操作-- 也就是 push to back(q.append(x)), peek/pop from front(q.pop(0)), size, 和 is empty 这些操作是合法的。
    你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。
"""


class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = []

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        self.q.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        length = len(self.q)
        while length-1 > 0:
            self.q.append(self.q.pop(0))
            length -= 1
        top = self.q.pop(0)
        return top

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.q[-1]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return not bool(self.q)

# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
