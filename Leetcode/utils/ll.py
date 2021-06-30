class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 用数组快速初始化链表
def InitLinkList(arr):
    if(len(arr) == 0):
        print("invalid input!")
        return None

    head = ListNode(arr[0])
    cur = head
    for a in arr[1:]:
        cur.next = ListNode(a)
        cur = cur.next
    return head


# 遍历打印链表，用->连接
def ForeachLinkList(head):
    cur = head
    while(cur):
        if(cur.next == None):
            print(cur.val)
        else:
            print("{}->".format(cur.val), end="")
        cur = cur.next


# 翻转链表
def ReverseLinkList(head):
    prev = None
    cur = head
    while(cur):
        pnext = cur.next
        cur.next = prev
        prev = cur
        cur = pnext
    return prev


# 链表节点数如果为奇数，则找到返回以中间点为头节点的链表；否则，则返回前半部分最后一个节点为头节点的链表
def FindMid(head):
    if(not head and not head.next):
        return head
    pf = head
    ps = head
    while(pf.next and pf.next.next): # 改为while(pf and pf.next)则偶数时返回后一个节点
        pf = pf.next.next
        ps = ps.next
    return ps


if __name__ == "__main__":
    l1 = InitLinkList([1,2,3,4])
    l2 = InitLinkList([1,2,3,4,5])



    res_rev = ReverseLinkList(l1)
    mid = FindMid(l2)

    ForeachLinkList(res_rev)
    ForeachLinkList(mid)
