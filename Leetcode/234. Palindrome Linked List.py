'''
Given the head of a singly linked list, return true if it is a palindrome.
Input: head = [1,2,2,1]
Output: true
Input: head = [1,2]
Output: false
'''

from utils.ll import *


# space O(n) time O(n)
def isPalindrome(head):
    """
    :type head: ListNode
    :rtype: bool
    """
    arr = []
    cur = head
    while(cur):
        arr.append(cur.val)
        cur = cur.next
    return arr == arr[::-1]


# space O(1) time O(n)
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


if __name__ == "__main__":
    head = InitLinkList([1, 2, 2, 1])
    # l1, l2 = isPalindrome_(head)
    # ForeachLinkList(l1)
    # ForeachLinkList(l2)
    print(isPalindrome_(head))
