#include <iostream>

using namespace std;

void swap(int&, int&);

int main7() {
	int a = 1;
	int b = 2;
	cout << "a=" << a << " " << "b=" << b << endl;
	cout << endl;
	swap(a, b);
	cout << "after swap,a&b:" << endl;
	cout << "a=" << a << " " << "b=" << b << endl;
	system("pause");
	return 0;
}

void swap(int& num1, int& num2) {
	int temp;
	temp = num1;
	num1 = num2;
	num2 = temp;	
}