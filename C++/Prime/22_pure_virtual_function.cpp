//#include <iostream>
//
//using namespace std;
//
//class Base {
//public:
//	virtual void func()=0;
//};
//
//class Son :public Base {
//public:
//	virtual void func() {
//		cout << "重写" << endl;
//	}
//};
//
//void test01() {
//	Base* base = new Son;//多态：父类的指针或引用指向子类对象
//	base->func();
//}
//
//int main() {
//	test01();
//
//
//	system("pause");
//	return 0;
//}