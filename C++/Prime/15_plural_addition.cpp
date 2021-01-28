////实现复数的直接打印和直接的加法；
//#include <iostream>
//
//using namespace std;
//
//class Plurality {
//	friend ostream& operator<<(ostream& cout, Plurality& p);
//	friend Plurality operator+(Plurality& p1, Plurality& p2);
//	friend Plurality operator-(Plurality& p1, Plurality& p2);
//public:
//	Plurality() {
//		p_a = 0;
//		p_b = 0;
//	}
//	Plurality(int a, int b) :p_a(a), p_b(b) {}
//
//private:
//	int p_a;
//	int p_b;
//};
//
//ostream& operator<<(ostream& cout, Plurality& p) {
//	//虚数部分为1或-1，不显示
//	if (p.p_b == 1) {
//		cout << p.p_a << " + i" << endl;
//	}
//	else if (p.p_b == -1) {
//		cout << p.p_a << " - i" << endl;
//	}
//	//虚数部分为负数，不显示运算符号，为了保证负号后空一格取反自添符号
//	else if (p.p_b < 0) {
//		cout << p.p_a <<" - "<< -1*p.p_b << "i" << endl;
//	}
//	//虚数为除1外的正数，添加正号输出
//	else {
//		cout << p.p_a << " + " << p.p_b << "i" << endl;
//	}
//	return cout;
//}
//
//Plurality operator+(Plurality& p1, Plurality& p2) {
//	Plurality temp;
//	temp.p_a = p1.p_a + p2.p_a;
//	temp.p_b = p1.p_b + p2.p_b;
//	return temp;
//}
//
//Plurality operator-(Plurality& p1, Plurality& p2) {
//	Plurality temp;
//	temp.p_a = p1.p_a - p2.p_a;
//	temp.p_b = p1.p_b - p2.p_b;
//	return temp;
//}
//
//void test() {
//	Plurality p1(1, 1);
//	Plurality p2(2, 5);
//	Plurality p3, p4;
//	p3 = p1 + p2;
//	p4 = p1 - p2;
//	cout << "p1 = " << p1 << endl;
//	cout << "p2 = " << p2 << endl;
//	cout << "p1 + p2 = " << p3 << endl;
//	cout << "p1 - p2 = " << p4 << endl;
//}
//
//int main() {
//	test();
//	system("pause");
//	return 0;
//}