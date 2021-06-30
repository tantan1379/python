#include "LinkList.h"
//初始化链表
struct ListNode* Init_LinkList() {
	ListNode* header = new ListNode(0);
	ListNode* pRear = header;
	float val = -1;
	while (1) {
		cout << "请输入链表的元素：" << endl;
		cin >> val;
		if (val == -1) {
			break;
		}
		pRear->next = new ListNode(val);
		pRear = pRear->next;
	}
	return header;
}

//在值为oldval的位置插入一个新的数据newval
void InsertByValue_LinkList(struct ListNode* header, float oldval, float newval) {
	if (nullptr == header) {
		return;
	}
	ListNode* pCurrent = header->next;
	ListNode* pPrev = header;
	while (pCurrent) {
		if (pCurrent->val == oldval) {
			break;
		}
		pPrev = pCurrent;
		pCurrent = pCurrent->next;
	}
	pPrev->next = new ListNode(newval, pCurrent);
}

//删除值为val的节点
void RemoveByValue_LinkList(struct ListNode* header, float delValue) {
	if (nullptr == header) {
		return;
	}
	ListNode* pCurrent = header->next;
	ListNode* pPrev = header;
	while (pCurrent) {
		if (pCurrent->val == delValue) {
			break;
		}
		pPrev = pCurrent;
		pCurrent = pCurrent->next;
	}
	if (nullptr == pCurrent) {
		cout << "Not Found!" << endl;
		return;
	}
	pPrev->next = pCurrent->next;
}

//遍历链表
void Foreach_LinkList(struct ListNode* header) {
	if (nullptr == header) {
		return;
	}
	ListNode* pCurrent = header->next;
	while (pCurrent) {
		cout << pCurrent->val << " ";
		cout << endl;
		pCurrent = pCurrent->next;
	}
}

//销毁链表
void Destroy_LinkList(struct ListNode* header) {
	if (nullptr == header) {
		return;
	}
	ListNode* pCurrent = header;
	while (pCurrent) {
		ListNode* pNext = pCurrent->next;
		cout << "销毁了节点" << pCurrent->val << endl;
		delete pCurrent;
		pCurrent = pNext;

	}
}

//清空链表
void Clear_LinkList(struct ListNode* header) {
	if (nullptr == header) {
		return;
	}
	ListNode* pCurrent = header->next;
	while (pCurrent) {
		ListNode* pNext = pCurrent->next;
		delete pCurrent;
		pCurrent = pNext;
	}
	header->next = nullptr;
}