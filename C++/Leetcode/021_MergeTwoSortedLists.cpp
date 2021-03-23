#include "ll.h"

ListNode* MergeTwoSortedLists(ListNode* l1, ListNode* l2) {
	ListNode* dummy = new ListNode(0);
	ListNode* pCurrent = dummy;
	if (l1 == nullptr) {
		return l2;
	}
	if (l2 == nullptr) {
		return l1;
	}
	while (l1 && l2) {
		if (l1->val > l2->val) {
			pCurrent->next = l2;
			l2 = l2->next;
			pCurrent = pCurrent->next;
		}
		else {
			pCurrent->next = l1;
			l1 = l1->next;
			pCurrent = pCurrent->next;
		}
	}
	if (l1) {
		pCurrent->next = l1;
	}
	if (l2) {
		pCurrent->next = l2;
	}
	ListNode* res = dummy->next;
	delete dummy;
	return res;
	//if (nullptr == l1) return l2;
	//else if (nullptr == l2) return l1;
	//else if (l1->val <= l2->val) {
	//	l1->next = MergeTwoSortedLists(l1->next, l2);
	//	return l1;
	//}
	//else {
	//	l2->next = MergeTwoSortedLists(l1, l2->next);
	//	return l2;
	//}
}


void test021() {
	vector<int>val1 = { 1, 2, 3 };
	vector<int>val2 = { 3, 4, 5 };
	ListNode* l1 = Init_LinkList(val1);
	Foreach_LinkList(l1);
	ListNode* l2 = Init_LinkList(val2);
	Foreach_LinkList(l2);
	ListNode* res = MergeTwoSortedLists(l1, l2);
	Foreach_LinkList(res);
}
