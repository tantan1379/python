## 链表基础

### 一、对比链表和数组：

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

### 二、链表概念：

链表是由一系列节点LinkNode组成，每个节点包含两个域：数据域（用于保存数据）和指针域（用于保存下一个节点的地址），ListNode在内存中是不可连续的。它包含一个指向**相同类型**数据结构的指针，因此可以说是一个包含对自身引用的类型。像这样的类型称为自引用数据类型或自引用数据结构。

可以分为：静态链表和动态链表
也可以分为：单向链表、双向链表、循环链表、单向循环链表、双向循环链表

### 三、链表结构：

![image-20210318102820213](C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210318102820213.png)

#### 1、头节点：

非空链表的第一个节点称为链表的头节点。要访问链表中的结点，需要有一个指向链表头的指针。从链表头开始，可以按照存储在每个结点中的后继指针访问链表中的其余结点。头节点的下一个节点我们通常称为第一个有效节点。

**Notes:**

1、获取到链表的第一个节点，就相当于获取整个链表。
2、头节点不包含任何有效数据。

#### 2、尾部节点：

每次链表增长或减短需要更新尾部节点的位置

##### Notes:

1、最后一个结点中的后继指针通常被设置为 nullptr 以指示链表的结束。
2、尾部节点通常被初始化为头节点

#### 3、哑节点：

在实际操作中，如果头节点需要进行返回往往需要考虑特殊情况，因此为了便于操作，通常引入哑节点
定义：`ListNode* dummy = new ListNode(0,header)`

##### Notes:

1、一般程序需要返回header，此时可以写作`return dummy->next; `如果有释放内存的需要，需要用一个临时变量寄存`dummy->next`，再释放；



### 四、单向链表的C++表示：

单向链表：只能通过前一个节点知道后一个元素的地址

#### 1、结构体定义：

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

#### 2、链表初始化：

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

#### 3、链表遍历（打印链表）：

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

#### 4、在指定值的位置插入新节点：

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



### 五、LeetCode:

#### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

迭代法：

```C++
class Solution{
public:
    ListNode* addTwoNumbers(ListNode* l1,ListNode* l2){
        ListNode* dummy = new ListNode(0);//创建哑节点（位于头节点前的节点，避免头节点的特殊判断）
        ListNode* pCurrent = dummy;//创建可供“移动”的当前节点
        int carry = 0;//进位
        while(l1||l2||carry){
            int target = (l1?l1->val:0)+(l2?l2->val:0)+carry;
            carry = target/10;//注：python用//
            ListNode* newNode = new ListNode(target%10);//创建新节点
            pCurrent->next = newNode;//连接到尾部节点的下一个节点
            pCurrent = pCurrent->next;//更新当前节点
            if(l1) l1 = l1->next;//更新两个相加链表
            if(l2) l2 = l2->next;
        }
        return dummy->next;//相当于return header
    }
};
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

```C++
ListNode* RemoveNthNodeFromEndofList(ListNode* header, int n) {
	if (nullptr == header->next && nullptr == header) {
		cout << "The linklist must have at least 2 elements!" << endl;
		exit(-1);
	}
	if (n <= 0) {
		cout << "n must be a positive number!" << endl;
		exit(-1);
	}
	ListNode* dummy = new ListNode(0, header);
	ListNode* fast = dummy;
	ListNode* slow = dummy;

	for (int i = 0; i <= n; i++) {
		fast = fast->next;
	}
	while (fast) {
		slow = slow->next;
		fast = fast->next;
	}
	slow->next = slow->next->next;

	ListNode* ans = dummy->next;
	delete dummy;
	return ans;
}
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

迭代法：

```C++
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(0);
    ListNode* pCurrent = &dummy;
    if(nullptr==l1) return l2;
    else if(nullptr==l2) return l1;
    else{
        while(l1&&l2){
            if(l1->val>l2->val){
                pCurrent->next = l2;
                l2 = l2->next;
                pCurrent = pCurrent->next;
            }
            else{
                pCurrent->next = l1;
                l1 = l1->next;
                pCurrent = pCurrent->next;
            }
   		}
    }
    pCurrent = l1? l1 : l2;
    return dummy->next;
}

```

递归法：

```C++
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if(nullptr==l1) return l2;
	else if(nullptr==l2) return l1;
	else if(l1->val>l2->val){
		l2->next = merTwoLists(l1,l2->next);
		return l2;
    else{
        l1->next = merTwoLists(l1->next,l2);
        return l1;
    }
}
```



#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

**你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]
```

迭代法：

```C++
ListNode* SwapNodesinPairs(ListNode* head) {
	if (nullptr == head || nullptr == head->next) {
		return head;
	}
	ListNode* dummy = new ListNode(0, head);
	ListNode* pCurrent = dummy;
	while (pCurrent->next && pCurrent->next->next) {
		ListNode* node1 = pCurrent->next;
		ListNode* node2 = pCurrent->next->next;
		node1->next = node2->next;
		pCurrent->next = node2;
		node2->next = node1;
		pCurrent = node1;
	}
	ListNode* res = dummy->next;
	delete dummy;
	return res;
}
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

```C++
ListNode* RotateList(ListNode* head, int k) {
	ListNode* oldrear = head;
	int n;
	for (n = 1; oldrear->next!=nullptr; n++) {//令n=1，此时循环结束后n即为链表长度
		oldrear = oldrear->next;
	}
	oldrear->next = head;
	ListNode* newrear = head;
	for (int i = 0; i < n - k %n - 1; i++) {//注意如果k大于链表长度时的写法
		newrear = newrear->next;
	}
	ListNode* newhead = newrear->next;
	newrear->next = nullptr;
	return newhead;
}
```



## 栈和队列

python中用**列表List**模拟栈和队列的功能： 
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

