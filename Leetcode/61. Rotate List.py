'''
Given the head of a linked list, rotate the list to the right by k places.
'''

from utils.ll import *


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


if __name__ == "__main__":
    head1 = InitLinkList([1, 2, 3, 4, 5])
    head2 = InitLinkList([1, 2, 3, 4, 5])
    ForeachLinkList(head1)
    res1 = rotateRight(head1, 3)
    ForeachLinkList(res1)
    res2 = rotateRight_(head2, 3)
    ForeachLinkList(res2)
