#include "ll.h"

//将两个链表按位相加
ListNode* AddTwoNumbers(ListNode* l1, ListNode* l2) {
	ListNode* dummy = new ListNode(0);
	ListNode* pCurrent = dummy;
	int carry = 0;
	while (l1 || l2 || carry) {
		int target = (l1 ? l1->val : 0) + (l2 ? l2->val : 0) + carry;
		carry = target / 10;
		pCurrent->next = new ListNode(target % 10);
		pCurrent = pCurrent->next;
		if (l1) l1 = l1->next;
		if (l2) l2 = l2->next;
	}
	ListNode* ans = dummy->next;
	delete dummy;
	return ans;
}

void test002() {
	vector<int> arr1 = { 1,2,3 };
	vector<int> arr2= { 5,6,7 };
	ListNode* l1 = Init_LinkList(arr1);
	ListNode* l2 = Init_LinkList(arr2);
	Foreach_LinkList(l1);
	Foreach_LinkList(l2);
	ListNode* target = AddTwoNumbers(l1, l2);
	Foreach_LinkList(target);
}