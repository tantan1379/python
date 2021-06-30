#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stack>
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
//初始化和遍历模板
ListNode* Init_LinkList(vector<int>& arr);
void Foreach_LinkList(ListNode* header);
//题目部分
void test002();
ListNode* AddTwoNumbers(ListNode* l1, ListNode* l2);
void test019();
ListNode* RemoveNthNodeFromEndofList(ListNode* header, int n);
void test021();
ListNode* MergeTwoSortedLists(ListNode* l1, ListNode* l2);
void test061();
ListNode* RotateList(ListNode* head, int k);
void test024();
ListNode* SwapNodesinPairs(ListNode* head);
void test143();
ListNode* reorderList(ListNode* head);
void test020();
bool inValid(string s);