'''
Given the head of a linked list, remove the nth node from the end of the list and return its head.
'''

from utils.ll import *


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


if __name__ == "__main__":
    head = InitLinkList([1, 2, 3, 4, 5, 6])
    res = removeNthFromEnd(head, 3)
    ForeachLinkList(res)
