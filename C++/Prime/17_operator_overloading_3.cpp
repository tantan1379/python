//#include <iostream>
//
//using namespace std;
//
//class Person {
//public:
//	Person(int age) {
//		m_Age = new int(age);
//	}	
//	int* m_Age;
//
//	~Person() {
//		if (m_Age != NULL) {
//			delete m_Age;
//			m_Age = NULL;
//		}
//	}
//
//	Person& operator=(Person& p) {
//		//±‡“Î∆˜Ã·π©£∫m_Age = p.m_Age;
//		if (m_Age != NULL) {
//			delete m_Age;
//			m_Age = NULL;
//		}
//		m_Age = new int(*p.m_Age);
//		return *this;
//	}
//};
//
//void test01() {
//	Person p1(18);
//	Person p2(20);
//	Person p3(30);
//
//	p3 = p2 = p1;
//
//	cout<<*p1.m_Age<<endl;
//	cout<<*p2.m_Age<<endl;
//	cout<<*p3.m_Age<<endl;
//}
//
//void test02() {
//	int* a = new int(10);
//	cout << a << endl;
//	cout << *a << endl;
//
//}
//
//int main() {
//	//test01();
//	test02();
//	system("pause");
//	return 0;
//}