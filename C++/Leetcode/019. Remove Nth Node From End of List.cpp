#include "ll.h"
#define _CRT_SECURE_NO_WARNINGS
//删除链表的倒数第N个节点
ListNode* RemoveNthNodeFromEndofList(ListNode* header, int n) {
	if (nullptr == header->next && nullptr == header) {
		cout << "The linklist must have at least 2 elements!" << endl;
		exit(-1);
	}
	if (n <= 0) {
		cout << "n must be a positive number!" << endl;
		exit(-1);
	}
	ListNode* dummy = new ListNode(0, header);
	ListNode* fast = dummy;
	ListNode* slow = dummy;

	for (int i = 0; i <= n; i++) {
		fast = fast->next;
	}
	while (fast) {
		slow = slow->next;
		fast = fast->next;
	}
	slow->next = slow->next->next;

	ListNode* ans = dummy->next;
	delete dummy;
	return ans;
}

void test019() {
	unsigned int delnum;
	vector<int> arr = {1,2,3};
	cout << "原链表为：";
	ListNode* header = Init_LinkList(arr);

	Foreach_LinkList(header);
	cout << "请输入你想倒数删除的节点数？" << endl;
	cin >> delnum;
	if (delnum <= arr.size()) {
		cout << "删除倒数第" << delnum << "个节点后：" << endl;
		ListNode* res = RemoveNthNodeFromEndofList(header, delnum);
		Foreach_LinkList(res);
	}
	else {
		cout << "数组越界！" << endl;
	}
	
}