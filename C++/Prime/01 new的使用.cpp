//#include <iostream>
//
//using namespace std;
//
//int* func() {
//	int* p = new int(10);
//	return p;
//}
//
//void test01() {
//	int* p = func();
//	cout << *p << endl;
//	delete p;
//}
//
//void test02() {
//	int *p = new int[10];
//	
//	for (int i = 0; i < 10; i++) {
//		p[i]= i;
//	}
//	for (int i = 0; i < 10; i++) {
//		cout << *(p+i) << endl;
//	}
//	delete[] p;
//}
//
//int main() {
//	test02();
//	system("pause");
//	return 0;
//}