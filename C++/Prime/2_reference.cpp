//#include <iostream>
//
//using namespace std;
//const int b = 30;
////值传递
//void test01(int a){
//
//}
//
////地址传递
//void test02(int* a){
//	*a = 30;
//}
//
////引用传递
//void test03(int &a){
//	a = 30;
//}
//
////返回静态变量（全局区）引用
//int& test04() {
//	static int a = 10;
//	return a;
//}
////返回局部变量的引用
////int& test05() {
////	int a = 10;
////	return a;
////}
//
//void showValue(const int a) {
//	cout << a << endl;
//}
//
//int main2() {
//	//int test = 10;
//	//test01(test);
//	//cout << test << endl;
//	//test02(&test);
//	//cout << test << endl;	
//	//test03(test);
//	//cout << test << endl;
//	//int& a = test04();
//	//cout << a << endl;
//	//cout << a << endl;	
//	//int& b = test05();
//	//cout << b << endl;
//	//cout << b << endl;
//	//int& a = test04();
//	//cout << a << endl;
//	//cout << a << endl;
//	//test04() = 1000;  //函数的调用可以作为左值
//	//cout << a << endl;  //相当于改变了a的别名，同时改变了a
//	//cout << a << endl;
//	showValue(20);
//	cout << b << endl;
//	return 0;
//}