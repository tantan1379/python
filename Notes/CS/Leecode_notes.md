## Leecode小记

#### 栈和队列
在python中用**列表List**模拟栈和队列的功能：   
栈在顶部先进后出（FILO），通常用List的最右端模拟栈的顶部   
**栈：**   
基本功能：  
    push(x) -- 元素 x 入栈 -> q.append(x)   
    pop() -- 移除栈顶元素,并返回该元素 -> return q.pop()   
    top() -- 获取栈顶元素 -> return q[0]   
    empty() -- 返回栈是否为空 -> return not bool(q)   
**队列：**   
基本功能：
    push(x) 将元素 x 推到队列的末尾 -> q.append(x)   
    pop() 从队列的开头移除并返回元素 -> return q.pop(0)   
    peek() 返回队列开头的元素 -> return q[0]   
    empty() -- 返回栈是否为空 -> return not bool(q)   

**用队列实现栈(L225)：**   
```
class MyStack(object):

    def __init__(self):
        self.q = []

    def push(self, x):
        self.q.append(x)

    def pop(self):
        length = len(self.q)
        while length-1 > 0:
            self.q.append(self.q.pop(0))
            length -= 1
        top = self.q.pop(0)
        return top

    def top(self):
        return self.q[-1]

    def empty(self):
        return not bool(self.q)

```    
**用栈实现队列(L232)：**   
```
class MyQueue(object):

    def __init__(self):
        self.q = []
        self.temp = []

    def push(self, x):
        self.q.append(x)

    def pop(self):
        length = len(self.q)
        while self.q:
            self.temp.append(self.q.pop())
        top = self.temp.pop()
        while length-1 > 0 :
            self.q.append(self.temp.pop())
            length-=1
        return top

    def peek(self):
        return self.q[0]

    def empty(self):
        return not bool(self.q)
```
