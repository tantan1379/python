#include <iostream>

using namespace std;

int main3()
{
	int temp = 0;
	int a[] = {9,8,7,6,5,4,3,2,1,0};
	int start = 0;
	int end = sizeof(a) / sizeof(int) - 1;
	while (start < end) {
		temp = a[end];
		a[end] = a[start];
		a[start] = temp;
		start++;
		end--;
	}
	for (int i = 0;i < (sizeof(a) / sizeof(int));i++) {
		cout<<a[i]<<endl;
	}
	
	system("pause");
	return 0;
}