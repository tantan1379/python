//#include <iostream>
//
//using namespace std;
//
//class Person {
//public:
//	static int m_A;
//	int m_B;
//	mutable int m_C;
//	//初始化列表对非静态成员变量进行初始化
//	Person() :m_B(12), m_C(14) {
//	}
//
//	Person(int m_B, int m_C) {
//		//this指针指向被调用的成员函数所属的对象
//		this->m_B = m_B;
//		this->m_C = m_C;
//	}
//
//	Person& PersonAdd(Person& p) {
//		this->m_B += p.m_B;
//		return *this;
//	}
//
//	//常函数
//	void set() const {
//		//成员属性声明时加关键字mutable后，在常对象和常函数中依然可以修改
//		m_C = 100;
//	}
//
//	//一般函数
//	void print() {
//		cout << "111" << endl;
//	}
//
//};
////静态成员变量在类内声明，类外初始化
//int Person::m_A = 100;
//
//void test1() {
//	Person p1;
//	//静态成员变量不占空间，若p1是一个空对象，编译器会分配一个直接用于区分空对象在内存的位置，每个空对象应该有一个独一的内存地址
//	cout << "对象p1占据的内存为：" << sizeof(p1) << endl;
//	cout << "m_A=" << p1.m_A << endl;
//	cout << "m_B=" << p1.m_B << endl;
//	cout << "m_C=" << p1.m_C << endl;
//	Person p2;
//	//所有对象共享同一份数据
//	p2.m_A = 200;
//	cout << "通过p2改变静态成员变量m_A后，p1的m_A="<<p1.m_A << endl;
//}
//
//void test2() {
//	Person p1(10, 12);
//	Person p2(10, 12);
//	//由于返回的是一个Person&类型的对象，因此每次都会返回p2的本体，而不是p2的拷贝
//	p2.PersonAdd(p1).PersonAdd(p1);//链式
//	cout << "p2经过两次和p1的PersonAdd后,m_B="<<p2.m_B << endl;
//}
//
//void test3() {
//	//常对象
//	const Person p3;
//	//p3.print(); //报错
//	//常对象只能调用常函数
//	p3.set();
//}
//
//int main() {
//	test1();
//	//test2();
//	//test3();
//	system("pause");
//	return 0;
//}