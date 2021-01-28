#include <iostream>

using namespace std;

void test();
int main4() {
	test();
	test();
	test();
	system("pause");
	return 0;
}

void test() {
	static int a=1;
	//a = 1;
	a++;
	cout << "a=" << a << endl;
}