//#define _CRT_SECURE_NO_WARNINGS
//#include <iostream>
//#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>
//
//using namespace std;
//
//struct ListNode {
//	float val;
//	ListNode* next;
//	ListNode() :val(0), next(nullptr) {}
//	ListNode(int x) : val(x), next(nullptr) {}
//	ListNode(int x, ListNode* next) : val(x), next(next) {}
//};
//
//int main() {
//	//static linked list
//	ListNode *node6 = new ListNode(60, nullptr);
//	ListNode *node5 = new ListNode(50, node6);
//	ListNode *node4 = new ListNode(40, node5);
//	ListNode *node3 = new ListNode(30, node4);
//	ListNode *node2 = new ListNode(20, node3);
//	ListNode *node1 = new ListNode(10, node2);
//
//	ListNode* pCurrent = node1;//定义辅助指针变量
//	while (pCurrent != nullptr) {
//		cout << pCurrent->val << " ";
//		pCurrent = pCurrent->next;//指针移动到下个节点的首地址
//	}
//	cout << endl;
//	system("pause");
//	return 0;
//}