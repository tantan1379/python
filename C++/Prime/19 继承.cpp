//#include <iostream>
//
//using namespace std;
//
//class Base {
//public:
//	int m_A = 200;
//	static int m_B;
//};
//int Base::m_B = 200;
//
//class Son :public Base
//{
//public:
//	int m_A = 100;
//	static int m_B;
//};
//int Son::m_B = 100;
//
//void test01() {
//	Son s;
//	cout << s.m_A << endl;
//	cout << s.Base::m_A << endl;
//}
//
//void test02() {
//	Son s;
//	//通过对象访问
//	cout << "通过对象访问" << endl;
//	cout << s.m_B << endl;
//	cout << s.Base::m_B << endl;
//	//通过类名访问
//	cout << "通过类名访问" << endl;
//	cout << Son::m_B << endl;
//	cout << Son::Base::m_B << endl;
//}
//
//
//int main() {
//	//test01();
//	test02();
//
//	system("pause");
//	return 0;
//}