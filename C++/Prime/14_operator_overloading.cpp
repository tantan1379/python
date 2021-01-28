//#include <iostream>
//
//using namespace std;
//
//class Person {
//public:
//	//声明全局函数operator+是Person的友元函数
//	friend Person operator+(Person& p1, Person& p2);
//	//全局函数重载<<运算符 (全局函数写在类内作为友元函数）
//	friend ostream& operator<<(ostream& cout, Person& p) {
//		return cout << "m_A = " << p.m_A << " m_B = " << p.m_B;
//	}
//	//利用初始化列表初始化属性
//	Person(int a, int b, int c, int d) :m_A(a), m_B(b), m_C(c), m_D(d) {};
//	////成员函数重载+运算符
//	//Person operator+(Person& p) {
//	//	Person temp;
//	//	temp.m_A = this->m_A + p.m_A;
//	//	temp.m_B = this->m_B + p.m_B;
//	//	return temp;
//	//}
//	int m_C;
//	int m_D;
//private:
//	int m_A;
//	int m_B;
//};
//
////全局函数重载+运算符
//Person operator+(Person& p1, Person& p2) {
//	Person temp(10, 10, 10, 10);
//	temp.m_A = p1.m_A + p2.m_A;
//	temp.m_B = p1.m_B + p2.m_B;
//	return temp;
//}
//
//
//
//void test1() {
//	Person p1(10, 10, 10, 10);
//	Person p2(10, 10, 10, 10);
//	Person p3 = p1 + p2;
//	cout << p3.m_C << endl;
//	cout << p3.m_D << endl;
//}
//
//void test2() {
//	Person p1(10, 10, 10, 10);
//	cout << p1 << endl;
//}
//
//int main() {
//	//test1();
//	test2();
//
//	system("pause");
//	return 0;
//}