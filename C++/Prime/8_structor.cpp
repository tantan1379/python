//#include <iostream>
//
//using namespace std;
//
//class Test {
//public:
//	int par = 10;
//
//	Test() {
//		cout << "默认构造函数调用" << endl;
//	}
//
//	~Test() {
//		cout << "析构函数调用" << endl;
//	}
//	
//	Test(int t) {
//		cout << "有参构造函数调用" << endl;
//	}
//
//	Test(const Test &t) {
//		cout << "拷贝构造函数调用" << endl;
//		par = t.par;
//	}
//};
//
//void doWork1(Test t) {
//}
//
//Test doWork2() {
//	Test t;
//	cout << "doWork中的t地址为" << (int*)&t << endl;//此处t的地址是利用默认构造函数创建的t的地址
//	return t;
//}
//
//void test0() {
//	Test a;
//	Test b(a);
//	cout << b.par << endl;
//}
//
////先调用拷贝构造函数创建一个t的副本用作函数的值传递，执行完doWork1后自动调用析构函数释放该副本，然后释放test1中的t
//void test1() {
//	Test t;
//	doWork1(t);
//}
//
////先调用拷贝构造函数创建一个t的副本用作值方式返回，返回值赋值结束后自动调用析构函数释放该副本，然后再释放test2中的t
//void test2() {
//	Test t = doWork2();
//	cout << "test2中的t地址为"<<(int*)&t << endl;//此处t的地址为利用拷贝构造函数创建t副本的地址
//}
//
//int main() {
//	//test0();//1、使用一个已经创建完毕的对象来初始化一个新对象
//	//test1();//2、值传递的方式给函数参数传值  
//	test2();//3、以值方式返回局部对象   
//	
//	//system("pause");
//}