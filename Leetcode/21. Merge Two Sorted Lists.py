"""
Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.
"""

from utils.ll import *


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


if __name__ == "__main__":
    l1 = InitLinkList([1, 2, 4])
    ForeachLinkList(l1)
    l2 = InitLinkList([1, 3, 4])
    ForeachLinkList(l2)
    res = mergeTwoLists(l1, l2)
    ForeachLinkList(res)
