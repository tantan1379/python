#include "ll.h"

ListNode* reorderList(ListNode* head) {
	if (!head || !head->next) return nullptr;
	ListNode* cur = head;
	vector<ListNode*> arr;
	while (cur) {
		arr.emplace_back(cur);
		cur = cur->next;
	}
	int i = 0, j = arr.size() - 1;
	while (i < j) {
		arr[i]->next = arr[j];
		i++;
		if (i == j) {
			break;
		}
		arr[j]->next = arr[i];
		j--;
	}
	arr[i]->next = nullptr;
	return head;
}

void test143() {
	vector<int> myarr{1,2,3,4,5,6};
	ListNode* head = Init_LinkList(myarr);
	Foreach_LinkList(head);
	ListNode* res = reorderList(head);
	Foreach_LinkList(res);
}