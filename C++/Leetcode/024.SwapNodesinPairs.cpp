#include "ll.h"
//迭代法（暴力）
ListNode* SwapNodesinPairs(ListNode* head) {
	if (nullptr == head || nullptr == head->next) {
		return head;
	}
	ListNode* dummy = new ListNode(0, head);
	ListNode* pCurrent = dummy;
	while (pCurrent->next && pCurrent->next->next) {
		ListNode* node1 = pCurrent->next;
		ListNode* node2 = pCurrent->next->next;
		node1->next = node2->next;
		pCurrent->next = node2;
		node2->next = node1;
		pCurrent = node1;
	}
	ListNode* res = dummy->next;
	delete dummy;
	return res;
}
//
////栈
//ListNode* SwapNodesinPairs_(ListNode* head) {
//	if (nullptr == head || nullptr == head->next) {
//		return head;
//	}
//	stack<ListNode*> stack;
//	ListNode* pCurrent = head;
//	ListNode* newhead = new ListNode();
//	while (pCurrent&&pCurrent->next) {
//		stack.push(pCurrent);
//		stack.push(pCurrent->next);
//		pCurrent = pCurrent->next->next;
//		newhead->next = stack.pop();
//		newhead->next->next = stack.pop();
//
//	}
//}

void test024() {
	vector<int>arr = {3,4,3,4,3,4};
	ListNode* header = Init_LinkList(arr);
	cout << "原始链表为：";
	Foreach_LinkList(header);
	cout << "两两交换后链表为：";
	header = SwapNodesinPairs(header);
	Foreach_LinkList(header);
}