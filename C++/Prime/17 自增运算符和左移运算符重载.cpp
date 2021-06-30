//#include <iostream>
//using namespace std;
//
//class MyInterger {
//	friend ostream& operator<<(ostream& cout, MyInterger myint);
//public:
//	MyInterger() {
//		m_Num = 0;
//	}
//
//    //返回引用类型，是为了实现一直对一个数据进行递增，否则是在对copy的对象进行递增
//	MyInterger& operator++() {
//		this->m_Num++;
//		//将自身做返回
//		return *this;
//	}
//
//	MyInterger operator++(int) {
//		MyInterger temp = *this;
//		this->m_Num++;
//		return temp;
//	}
//
//	MyInterger& operator--() {
//		this->m_Num--;
//		//将自身做返回
//		return *this;
//	}
//
//	MyInterger operator--(int) {
//		MyInterger temp = *this;
//		this->m_Num--;
//		return temp;
//	}
//
//
//private:
//	int m_Num;
//};
//
//ostream& operator<<(ostream& cout, MyInterger myint) {
//	cout << myint.m_Num;
//	return cout;
//}
//
//void test1() {
//	MyInterger myint;
//
//	cout << "++myint = " << ++myint << endl;
//	cout << "myint = " << myint << endl;
//	cout << "myint++ = "<<myint++ << endl;
//	cout << "myint = "<<myint << endl;
//}
//
//void test2() {
//	MyInterger myint;
//
//	cout << "--myint = " << --myint << endl;
//	cout << "myint = " << myint << endl;
//	cout << "myint-- = "<<myint-- << endl;
//	cout << "myint = "<<myint << endl;
//}
//
//int main() {
//	//test1();
//	test2();
//
//	system("pause");
//	return 0;
//}