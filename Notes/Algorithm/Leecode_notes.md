## 基础

我们从算法所占用的「时间」和「空间」两个维度考察算法之间的优劣。

- 时间维度：是指执行当前算法所消耗的时间，我们通常用「时间复杂度」来描述。
- 空间维度：是指执行当前算法需要占用多少内存空间，我们通常用「空间复杂度」来描述。

我们通常使用「 **大O符号表示法** 」描述时间的复杂度，该符号又称为**渐进符号**。

用渐进符号可以将复杂度分为常数阶O(1)、线性阶O(n)、指数阶O(2^n)、对数阶O(logn)、线性对数阶O(nlogn)

大O复杂度曲线：

![image-20210410115713666](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210410115713666.png)

抽象数据结构操作复杂度：

![image-20210410115804432](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210410115804432.png)

数组排序

![image-20210410115822169](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210410115822169.png)

## 链表

### 一、原理：

**对比数组和链表：**

**数组：**
数组的优点：
各个元素的内存空间的地址连续，因此可以通过位置快速访问、定位某个元素。
数组的缺点：
（1）修改数组（删除、插入）将导致元素大量移动，效率极低；（2）静态数组的内存空间可能存在空间不足或空间浪费问题。

**链表：**
链表的优点；
链表在指定位置进行插入和删除操作时，只需要修改被删节点上一节点的链接地址，不需要移动元素。
链表的缺点：
（1）相对于数组，多了指针域的内存开销；（2）查找效率较低。

**概念：**

链表是由一系列节点LinkNode组成，每个节点包含两个域：数据域（用于保存数据）和指针域（用于保存下一个节点的地址），ListNode在内存中是不可连续的。它包含一个指向**相同类型**数据结构的指针，因此可以说是一个包含对自身引用的类型。像这样的类型称为自引用数据类型或自引用数据结构。

可以分为：静态链表和动态链表
也可以分为：单向链表、双向链表、循环链表、单向循环链表、双向循环链表

**结构：**

![image-20210318102820213](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210318102820213.png)

**1、头节点：**

非空链表的第一个节点称为链表的头节点。要访问链表中的结点，需要有一个指向链表头的指针。从链表头开始，可以按照存储在每个结点中的后继指针访问链表中的其余结点。头节点的下一个节点我们通常称为第一个有效节点。

**Notes:**

1、获取到链表的第一个节点，就相当于获取整个链表。
2、头节点不包含任何有效数据。

**2、尾部节点：**

每次链表增长或减短需要更新尾部节点的位置

**Notes:**

1、最后一个结点中的后继指针通常被设置为 nullptr 以指示链表的结束。
2、尾部节点通常被初始化为头节点

**3、哑节点：**

在实际操作中，如果头节点需要进行返回往往需要考虑特殊情况，因此为了便于操作，通常引入哑节点
定义：`ListNode* dummy = new ListNode(0,header)`

**Notes:**

1、一般程序需要返回header，此时可以写作`return dummy->next; `如果有释放内存的需要，需要用一个临时变量寄存`dummy->next`，再释放；



### 二、程序实现(c++)

单向链表：只能通过前一个节点知道后一个元素的地址

1、结构体定义：

```c++
    struct ListNode
    {
        float val;
        ListNode *next;
        //构造函数
        ListNode():val(0),next(nullptr){}
        ListNode(float x):val(x),next(nullptr){}
        ListNode(float x,ListNode* next):val(x),next(next){}
    };
```

2、链表初始化：

```C++
ListNode* Init_LinkList(vector<int>& arr) {//引用传递
	if (arr.size() == 0) {
		cout << "警告，链表为空！" << endl;
		return nullptr;
	}
	ListNode* header = new ListNode(arr[0]);
	ListNode* pCurrent = header;
	for (int i = 1; i < arr.size(); i++) {
		pCurrent->next = new ListNode(arr[i]);
		pCurrent = pCurrent->next;
	}
	return header;
}
```

3、链表遍历（打印链表）：

```C++
void Foreach_LinkList(ListNode* header) {
	if (nullptr == header) {
		cout << "This is an empty LinkList" << endl;
		return;
	}
	ListNode* pCurrent = header->next;
	while (pCurrent) {
		cout << pCurrent->val << " ";
		pCurrent = pCurrent->next;
	}
}
```

4、在指定值的位置插入新节点：

```C++
void InsertByValue_LinkList(ListNode* header, float oldval, float newval) {
	if (nullptr == header) {
		cout << "This is an empty LinkList" << endl;
		return;
	}
	//创建两个辅助指针变量
	ListNode* pPrev = header;
	ListNode* pCurrent = header->next;
	while (pCurrent) {
		if (pCurrent->val == oldval) {
			break;
		}
		pPrev = pPrev->next;
		pCurrent = pCurrent->next;
	}
	//如果pCurrent为NULL则说明链表中不存在值为oldval的节点，插入到链表的尾部
	pPrev->next = new ListNode(newval, pCurrent);//pPrev->next = newNode; newNode->next = pCurrent;
}
```



### 三、LeetCode:

#### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    cur = dummy
    carry = 0
    while l1 or l2 or carry:
        target = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
        carry = target // 10
        cur.next = ListNode(target % 10)
        cur = cur.next
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    return dummy.next
```



#### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
输入：head = [1], n = 1
输出：[]
输入：head = [1,2], n = 1
输出：[1]
```

快慢指针：

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    slow = dummy
    fast = head
    for _ in range(n):
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next
```



#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
输入：l1 = [], l2 = []
输出：[]
输入：l1 = [], l2 = [0]
输出：[0]
```

```python
def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    cur = dummy
    while(l1 and l2):
        if(l1.val >= l2.val):
            cur.next = l2
            l2 = l2.next
            cur = l2
        else:
            cur.next = l1
            l1 = l1.next
            cur = l1

    cur.next = l2 if(l1 == None) else l1
    return dummy.next
```



#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

**你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]
```

迭代法：

```python
def swapPairs(head: ListNode):
    dummy = ListNode(0, head)
    cur = dummy
    while(cur.next and cur.next.next):
        node1 = cur.next
        node2 = cur.next.next
        cur.next = node2
        node1.next = node2.next
        node2.next = node1
        cur = node1
    return dummy.next
```



#### [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

给定一个链表，旋转链表，将链表每个节点向右移动 *k* 个位置，其中 *k* 是非负数。

```
输入: 0->1->2->NULL, k = 4
输出: 2->0->1->NULL
解释:
向右旋转 1 步: 2->0->1->NULL
向右旋转 2 步: 1->2->0->NULL
向右旋转 3 步: 0->1->2->NULL
向右旋转 4 步: 2->0->1->NULL
```

开闭环法：

```python
def rotateRight(head, k):
    if(head == None or head.next == None):
        return head
    cur = head
    n = 1
    while(cur.next):
        cur = cur.next
        n = n+1
    cur.next = head
    cur = head
    k = n-k % n-1
    for _ in range(k):
        cur = cur.next
    newhead = cur.next
    cur.next = None
    return newhead
```

快慢指针法：

```python
def rotateRight_(head, k):
    if not head or not head.next:
        return head
    cur, slow, fast = head, head, head
    n = 0
    while(cur):
        cur = cur.next
        n += 1
    if not k or not k % n:
        return head
    for _ in range(k % n):
        fast = fast.next
    while(fast.next):
        slow = slow.next
        fast = fast.next
    newhead = slow.next
    slow.next = None
    fast.next = head
    return newhead
```



#### [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

给定一个单链表 *L*：*L*0→*L*1→…→*L**n*-1→*L*n ，
 将其重新排列后变为： *L*0→*L**n*→*L*1→*L**n*-1→*L*2→*L**n*-2→…

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

```
给定链表 1->2->3->4, 重新排列为 1->4->2->3.
给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
```

线性表：

```python
def reorderList(head):
    if(not head or not head.next):
        return head
    newlist = []
    cur = head
    while(cur):
        newlist.append(cur)
        cur = cur.next
    i, j = 0, len(newlist)-1
    while i < j:
        newlist[i].next = newlist[j]
        i += 1
        if i == j:
            break
        newlist[j].next = newlist[i]
        j -= 1
    newlist[i].next = None
    return head
```

迭代法：

```python
def reorderList_(head):
    if(not head or not head.next):
        return head
    first_end = FindMid(head)
    l2 = ReverseLinkList(first_end.next)
    first_end.next = None
    l1 = head
    while(l1 and l2):
        temp1 = l1.next
        temp2 = l2.next
        l1.next = l2
        l1 = temp1
        l2.next = l1
        l2 = temp2
    if(not l1 and l2):
        l2.next = None
    if(not l2 and l1):
        l1.next = None
    return head
```



#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

反转一个单链表。

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

```python
def reverseList(head):
    if(not head or not head.next):
        return head
    prev = None
    cur = head
    while(cur):
        pnext = cur.next
        cur.next = prev
        prev = cur
        cur = pnext
    return prev
```



#### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

请判断一个链表是否为回文链表。

```
输入: 1->2
输出: false


输入: 1->2->2->1
输出: true
```

线性表：

```python
def isPalindrome(head):
    arr = []
    cur = head
    while(cur):
        arr.append(cur.val)
        cur = cur.next
    return arr == arr[::-1]
```

拆分链表、反转链表、比较链表：

```python
def isPalindrome_(head):
    if not head:
        return True
    first_end = FindMid(head)
    second_start = first_end.next
    l2 = ReverseLinkList(second_start)
    first_end.next = None
    l1 = head
    while(l1 and l2):
        if(l1.val == l2.val):
            l1 = l1.next
            l2 = l2.next
        else:
            return False
    return True
```

#### [面试题 02.03. 删除中间节点](https://leetcode-cn.com/problems/delete-middle-node-lcci/)

实现一种算法，删除单向链表中间的某个节点（即不是第一个或最后一个节点），假定你只能访问该节点。

```
输入：单向链表a->b->c->d->e->f中的节点c
结果：不返回任何数据，但该链表变为a->b->d->e->f
```

```python
def deleteNode(node):
    node.val = node.next.val # 将下一个节点的值复制到删除的节点处
    node.next = node.next.next # 删除下一个节点
```





## 哈希表

散列表（Hash table），是根据关键码值(Key value)而直接进行访问的数据结构。也就是说，它通过把关键码值映射到表中一个位置来访问记录，以加快**查找**的速度。这个映射函数叫做散列函数，存放记录的数组叫做散列表。

### 一、原理：

**存储位置=f(关键字)**

哈希表hashtable(key，value) 就是把Key通过一个固定的算法函数既所谓的哈希函数转换成一个整型数字，然后就将该数字对数组长度进行取余，**取余结果就当作数组的下标**，将value存储在以该数字为下标的数组空间里。（或者：把任意长度的输入（又叫做预映射， pre-image），通过散列算法，变换成固定长度的输出，该输出就是散列值。这种转换是一种压缩映射，也就是，散列值的空间通常远小于输入的空间，**不同的输入可能会散列成相同的输出**，而不可能从散列值来唯一的确定输入值。简单的说就是一种将任意长度的消息压缩到某一固定长度的消息摘要的函数。）
而当使用哈希表进行查询的时候，就是再次使用哈希函数将key转换为对应的数组下标，并定位到该空间获取value，



**优缺点：**

优点：不论哈希表中有多少数据，查找、插入、删除（有时包括删除）只需要接近常量的时间即0(1）的时间级。实际上，这只需要几条机器指令。
哈希表运算得非常快，在计算机程序中，如果需要在一秒种内查找上千条记录通常使用哈希表（例如拼写检查器)哈希表的速度明显比树快，树的操作通常需要O(N)的时间级。哈希表不仅速度快，编程实现也相对容易。

如果不需要有序遍历数据，并且可以提前预测数据量的大小。那么哈希表在速度和易用性方面是无与伦比的。

缺点：它是基于数组的，数组创建后难于扩展，某些哈希表被基本填满时，性能下降得非常严重，所以程序员必须要清楚表中将要存储多少数据（或者准备好定期地把数据转移到更大的哈希表中，这是个费时的过程）。



### 二、LeetCode

#### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

```python
def twoSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    mapping = {}
    for i in range(len(nums)):
        diff = target-nums[i]
        if diff not in mapping.keys():
            mapping[nums[i]] = i
        else:
            return [mapping[diff],i]
```







## 栈和队列

python中用**列表List**模拟栈和队列的功能： 
栈在顶部先进后出（FILO），通常用List的最右端模拟栈的顶部 

### 一、栈： 

基本功能：
  push(x) -- 元素 x 入栈 -> q.append(x) 
  pop() -- 移除栈顶元素,并返回该元素 -> return q.pop() 
  top() -- 获取栈顶元素 -> return q[0] 
  empty() -- 返回栈是否为空 -> return not bool(q) 



### 二、队列： 

基本功能：
  push(x) 将元素 x 推到队列的末尾 -> q.append(x) 
  pop() 从队列的开头移除并返回元素 -> return q.pop(0) 
  peek() 返回队列开头的元素 -> return q[0] 
  empty() -- 返回栈是否为空 -> return not bool(q) 



### 三、LeetCode

#### [225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通队列的全部四种操作（`push`、`top`、`pop` 和 `empty`）。

实现 `MyStack` 类：

- `void push(int x)` 将元素 x 压入栈顶。
- `int pop()` 移除并返回栈顶元素。
- `int top()` 返回栈顶元素。
- `boolean empty()` 如果栈是空的，返回 `true` ；否则，返回 `false` 。

```python
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
#### [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（`push`、`pop`、`peek`、`empty`）：

实现 `MyQueue` 类：

- `void push(int x)` 将元素 x 推到队列的末尾
- `int pop()` 从队列的开头移除并返回元素
- `int peek()` 返回队列开头的元素
- `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`

```python
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



## 堆

### 一、原理

**堆(heap)**是一种**完全二叉树**。堆是具有以下性质：每个结点的值都大于或等于其左右孩子结点的值，称为**大顶堆**；或者每个结点的值都小于或等于其左右孩子结点的值，称为**小顶堆**。我们对堆中的结点按层进行编号，如下图所示。

![image-20210410143218686](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210410143218686.png)

**完全二叉树：**

叶子结点只能出现在最下层和次下层，且最下层的叶子结点集中在树的左部。（生成顺序：从上到下，从左到右）

**构建堆的方法（heapify)：**

a.将无需序列构建成一个堆，根据升序降序需求选择大顶堆或小顶堆;
b.将堆顶元素与末尾元素交换，将最大元素"沉"到数组末端;
c.重新调整结构，使其满足堆定义，然后继续交换堆顶元素与当前末尾元素，反复执行调整+交换步骤，直到整个序列有序

**编程实现：**

我们可以用一个**数组**表示完全二叉树，用**数组下标**表示二叉树的位置。
如果二叉树的位置为i，则其父节点的位置为**floor((i-1)/2) **，其子节点的位置分别为**2i+1**和**2i+2**.



### 二、程序实现

```python
def heapify(arr, n, i): 
    largest = i  
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
    if l < n and arr[i] < arr[l]: 
        largest = l 
    if r < n and arr[largest] < arr[r]: 
        largest = r 
    if largest != i: 
        arr[i],arr[largest] = arr[largest],arr[i]  # 交换
        heapify(arr, n, largest) 

def heapSort(arr):
    heapSize = len(arr)
    for i in range((heapSize-1)//2,-1,-1):  # 创建大顶堆
        heapify(arr,heapSize,i)
    for i in range(heapSize-1,0,-1):  # 每次都将堆中最大的元素放在堆的末尾，每次heapify的规模-1
        arr[i],arr[0]=arr[0],arr[i]
        heapify(arr,i,0)
```







## Copy

#### 链表

```python
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
# 遍历打印链表，用->连接
def ForeachLinkList(head):
    cur = head
    while(cur):
        if(cur.next == None):
            print(cur.val)
        else:
            print("{}->".format(cur.val), end="")
        cur = cur.next

# 链表节点数如果为奇数，则找到返回以中间点为头节点的链表；否则，则返回前半部分最后一个节点为头节点的链表        
def FindMid(self,head):
    if(not head and not head.next):
        return head
    pf = head
    ps = head
    while(pf.next and pf.next.next):
        pf = pf.next.next
        ps = ps.next
    return ps

# 翻转链表，返回新链表的表头
def ReverseLinkList(self,head):
    prev = None
    cur = head
    while(cur):
        pnext = cur.next
        cur.next = prev
        prev = cur
        cur = pnext
    return prev
```

#### 堆

```python
# 堆排序
def heapify(arr, n, i): 
    largest = i  
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
    if l < n and arr[i] < arr[l]: 
        largest = l 
    if r < n and arr[largest] < arr[r]: 
        largest = r 
    if largest != i: 
        arr[i],arr[largest] = arr[largest],arr[i]  # 交换
        heapify(arr, n, largest) 

def heapSort(arr):
    heapSize = len(arr)
    for i in range((heapSize-1)//2,-1,-1):  # 创建大顶堆
        heapify(arr,heapSize,i)
    for i in range(heapSize-1,0,-1):  # 每次都将堆中最大的元素放在堆的末尾，每次heapify的规模-1
        arr[i],arr[0]=arr[0],arr[i]
        heapify(arr,i,0)
```

