#include <stdio.h>
#include <iostream>

using namespace std;


void heapify(int tree[], int n, int i) { // n表示完全二叉树的总节点数，i表示对某节点做heapify操作
	if (i >= n) {
		return;
	}
	int c1 = 2 * i + 1;
	int c2 = 2 * i + 2;
	int max = i;
	if (c1 < n && tree[c1] > tree[max]) {
		max = c1;
	}
	if (c2 < n && tree[c2] > tree[max]) {
		max = c2;
	}
	if (max != i) {
		swap(tree[max], tree[i]);
		heapify(tree, n, max);
	}
}

void heapSort(int tree[], int n) {
	for (int i = (n - 1)/2; i >= 0; i--) { //创建大顶堆
		heapify(tree, n, i);
	}
	for (int i = n - 1; i > 0; i--) {  //每次都将堆中最大的元素放在堆的末尾，每次heapify的规模-1
		swap(tree[i], tree[0]);
		heapify(tree, i, 0);
	}
}

int main() {
	int tree[] = { 2,5,3,1,10,4 };
	int n = 6;
	for (const auto& x : tree) cout << x << " ";
	cout << endl;
	heapSort(tree, n);
	for (const auto& x : tree) cout << x << " ";
	return 0;
}
 