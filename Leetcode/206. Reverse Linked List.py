'''
Given the head of a singly linked list, reverse the list, and return the reversed list.
'''

from utils.ll import *


def reverseList(head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
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


if __name__ == '__main__':
    headarr = [1, 2, 3, 4, 5, 6, 7, 8]
    head = InitLinkList(headarr)
    ForeachLinkList(head)
    newhead = reverseList(head)
    ForeachLinkList(newhead)
    ForeachLinkList(head)
