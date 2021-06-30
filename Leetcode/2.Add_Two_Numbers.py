"""

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order,
 and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.
给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。如果，我们将这两个数相
加起来，则会返回一个新的链表来表示它们的和。
您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
"""
from utils.ll import *


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


if __name__ == "__main__":
    ll1 = InitLinkList([3, 4, 5])
    ForeachLinkList(ll1)
    ll2 = InitLinkList([5, 6, 7])
    ForeachLinkList(ll2)
    res = addTwoNumbers(ll1, ll2)
    ForeachLinkList(res)
