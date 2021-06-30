'''
You are given the head of a singly linked-list. The list can be represented as:
L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.
'''

from utils.ll import *


# space O(n) time O(n)
def reorderList(head):
    """
    :type head: ListNode
    :rtype: None Do not return anything, modify head in-place instead.
    """
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


# fast and slow pointer
def reorderList_(head):
    """
    :type head: ListNode
    :rtype: None Do not return anything, modify head in-place instead.
    """
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


if __name__ == "__main__":
    myarr = [1, 2, 3, 4, 5]
    head = InitLinkList(myarr)
    ForeachLinkList(head)
    # newhead1,newhead2 = reorderList_(head)
    # ForeachLinkList(newhead1)
    # ForeachLinkList(newhead2)
    res = reorderList_(head)
    ForeachLinkList(res)
