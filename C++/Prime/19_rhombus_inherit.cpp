//#include <iostream>
//
//using namespace std;
//
//class Animal {
//public:
//	int m_Age;
//};
//
//class Sheep :virtual public Animal {};
//
//class Camel :virtual public Animal {};
//
//class Alpaca :public Sheep, public Camel {};
//
//void test01() {
//	Alpaca al;
//	al.Sheep::m_Age = 18;
//	al.Camel::m_Age = 28;
//	cout << al.Sheep::m_Age << endl; 
//	cout << al.Camel::m_Age << endl;
//	cout << al.m_Age << endl;
//}
//
//int main() {
//	test01();
//	system("pause");
//	return 0;
//}