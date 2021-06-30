"""
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the
functions of a normal queue (push, peek, pop, and empty).

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列的支持的所有操作（push、pop、peek、empty）：

实现 MyQueue 类：

    void push(int x) 将元素 x 推到队列的末尾
    int pop() 从队列的开头移除并返回元素
    int peek() 返回队列开头的元素
    boolean empty() 如果队列为空，返回 true ；否则，返回 false

说明：

    你只能使用标准的栈操作 —— 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
    你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

"""


class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = []
        self.temp = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        # self.q.append(x)
        while self.q:
            self.temp.append(self.q.pop())
        self.temp.append(x)
        while self.temp:
            self.q.append(self.temp.pop)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        # length = len(self.q)
        # while self.q:
        #     self.temp.append(self.q.pop())
        # top = self.temp.pop()
        # while length >1:
        #     self.q.append(self.temp.pop())
        # return top
        return self.q.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.q[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return not bool(self.q)

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
