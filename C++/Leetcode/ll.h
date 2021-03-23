#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
using namespace std;
struct ListNode {
public:
	int val;
	ListNode* next;
	ListNode(int x, ListNode* next) :val(x), next(next) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode() :val(0), next(nullptr) {}
};
ListNode* Init_LinkList(vector<int>& arr);
void Foreach_LinkList(ListNode* header);
void test002();
ListNode* AddTwoNumbers(ListNode* l1, ListNode* l2);
void test019();
ListNode* RemoveNthNodeFromEndofList(ListNode* header, int n);
void test021();
ListNode* MergeTwoSortedLists(ListNode* l1, ListNode* l2);
