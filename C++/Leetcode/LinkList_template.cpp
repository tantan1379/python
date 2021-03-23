#include "ll.h"
#include <iostream>
#include <vector>

using namespace std;

//初始化链表
ListNode* Init_LinkList(vector<int>& arr) {
	if (arr.size() == 0) {
		return 0;
	}
	ListNode* header = new ListNode(arr[0]);
	ListNode* pCurrent = header;
	for (int i = 1; i < arr.size(); i++) {
		pCurrent->next = new ListNode(arr[i]);
		pCurrent = pCurrent->next;
	}
	return header;

	//ListNode* header = new ListNode(1);
	//ListNode* pRear = header;
	//int val = -1;
	//while (1) {
	//	cout << "请输入链表的元素：";
	//	cin >> val;
	//	if (val == -1) {
	//		break;
	//	}
	//	pRear->next = new ListNode(val);
	//	pRear = pRear->next;
	//}
	//return header;
}

//遍历链表
void Foreach_LinkList(struct ListNode* header) {
	if (nullptr == header) {
		return;
	}
	ListNode* pCurrent = header;
	while (pCurrent) {
		if (pCurrent->next != nullptr) {
			cout << pCurrent->val << "->";
	}
		else {
			cout << pCurrent->val;
		}
		pCurrent = pCurrent->next;
	}
	cout << endl;
}