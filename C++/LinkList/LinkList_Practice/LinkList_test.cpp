#define _CRT_SECURE_NO_WARNINGS
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "LinkList.h"

using namespace std;

void test1() {
	ListNode* header = Init_LinkList();
	Foreach_LinkList(header);
	InsertByValue_LinkList(header, 3, 3.5);
	cout << "-------------" << endl;
	Foreach_LinkList(header);
	RemoveByValue_LinkList(header, 4);
	cout << "-------------" << endl;
	Foreach_LinkList(header);
	Clear_LinkList(header);
	cout << "-------------"<<endl;
	Foreach_LinkList(header);
	InsertByValue_LinkList(header, 10, 11);
	InsertByValue_LinkList(header, 10, 21);
	InsertByValue_LinkList(header, 10, 31);
	InsertByValue_LinkList(header, 10, 41);
	Foreach_LinkList(header);
	cout << "-------------" << endl;
	Destroy_LinkList(header);
}

void test2() {
	ListNode* header = Init_LinkList();
	cout << "-------------" << endl;
	Foreach_LinkList(header);
	cout << "-------------" << endl;
	InsertByValue_LinkList(header, 3, 10);
	Foreach_LinkList(header);
	cout << "-------------" << endl;
	RemoveByValue_LinkList(header, 6);
	Foreach_LinkList(header);
	cout << "-------------" << endl;
	Clear_LinkList(header);
	Foreach_LinkList(header);
	cout << "-------------" << endl;
	Destroy_LinkList(header);
}
//
//int main() {
//	test2();
//
//	system("pause");
//	return 0;
//}