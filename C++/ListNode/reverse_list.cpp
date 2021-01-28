# include <stdio.h>
# include <iostream>

using namespace std;

struct ListNode {
	int val;
	ListNode* next;
	ListNode(int x) :val(x), next(NULL) {}
};

class Solution {
public:
	ListNode* reverseList(ListNode* head) {
		ListNode* new_head = NULL;
		while (head) {
			ListNode* next = head->next;
			head->next = new_head;
			new_head = head;
			head = next;
		}
		return new_head;
	}
};


int main()
{
	system("pause");
	return 0;
}
