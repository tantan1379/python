#pragma once
#include <stdlib.h>
#include <iostream>
using namespace std;
//定义节点数据类型
struct ListNode {
	float val;
	ListNode* next;
	ListNode(float x) :val(x), next(nullptr) {}
	ListNode() :val(0), next(nullptr) {}
	ListNode(float x, ListNode* next) :val(x), next(next) {}
};
//初始化链表
struct ListNode* Init_LinkList();
//在值为oldval的位置插入一个新的数据newval
void InsertByValue_LinkList(struct ListNode* header, float oldval, float newval);
//删除值为val的节点
void RemoveByValue_LinkList(struct ListNode* header, float delValue);
//遍历链表
void Foreach_LinkList(struct ListNode* header);
//销毁链表
void Destroy_LinkList(struct ListNode* header);
//清空链表
void Clear_LinkList(struct ListNode* header);