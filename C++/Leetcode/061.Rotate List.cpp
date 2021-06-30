#include "ll.h"

ListNode* RotateList(ListNode* head, int k) {
	ListNode* oldrear = head;
	int n;
	for (n = 1; oldrear->next!=nullptr; n++) {
		oldrear = oldrear->next;
	}
	oldrear->next = head;
	ListNode* newrear = head;
	for (int i = 0; i < n - k %n - 1; i++) {
		newrear = newrear->next;
	}
	ListNode* newhead = newrear->next;
	newrear->next = nullptr;
	return newhead;
}

void test061() {
	int step;
	vector<int> arr = {1,2,3,4,5};
	ListNode* head = Init_LinkList(arr);
	cout << "原始链表为：";
	Foreach_LinkList(head);
	cout << "请输入移动的步数：";
	cin >> step;
	ListNode* newhead = RotateList(head, step);
	cout << "旋转" << step << "步后链表为：";
	Foreach_LinkList(newhead);
}