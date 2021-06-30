'''
Given a linked list, swap every two adjacent nodes and return its head.
'''

from utils.ll import *


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


if __name__ == "__main__":
    head = InitLinkList([1, 2, 3, 4, 5])
    res = swapPairs(head)
    ForeachLinkList(res)
