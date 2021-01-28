//#include <iostream>
//
//using namespace std;
//
//class Person {
//public:
//	int m_age;
//	int* m_height;
//
//	Person(int age,int height) {
//		m_age = age;
//		m_height = new int(height);
//		cout << "有参构造函数" << endl;
//	}
//
//	Person(const Person& p) {
//		m_age = p.m_age;
//		m_height = new int(*p.m_height);//如果用默认拷贝构造函数，则m_height和拷贝的对象的m_height指向同一个地址
//		cout << "拷贝构造函数" << endl;
//	}
//
//	~Person() {
//		if (m_height != NULL) {
//			delete m_height;
//			m_height = NULL;
//		}
//		cout << "析构函数调用" << endl;
//	}
//};
//
//void test01() {
//	Person p1(18, 160);
//	cout << (int)p1.m_height << endl;
//	cout << p1.m_age << " " << *p1.m_height << endl;
//	Person p2(p1);//由于后入栈的先出栈，所以p2先被delete,然后p1才被delete
//	cout << (int)p2.m_height << endl;
//	cout << p2.m_age << " " << *p2.m_height << endl;
//}
//
//int main() {
//	test01();
//
//
//	system("pause");
//	return 0;
//}