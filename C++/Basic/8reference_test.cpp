#include <iostream>

using namespace std;

int main8() {
	int a[] = { 1,2,3 };
	int (&b)[3] = a;
	
	cout << a[2] << endl;
	cout << b[2] << endl;
	system("pause");
	return 0;
}