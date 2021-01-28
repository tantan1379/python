#include <iostream>

using namespace std;

////创建立方体类
////设计属性和行为，属性包括长宽高，行为包括获取立方体的面积和体积
////分别利用全局函数和成员函数判断两个立方体是否相等
//
//class Cube {
//private:
//	int m_H;
//	int m_L;
//	int m_W;
//public:
//	int get_H() {
//		return m_H;
//	}
//
//	int get_L() {
//		return m_L;
//	}
//
//	int get_W() {
//		return m_W;
//	}
//
//	void set_m_H(int H){
//		m_H = H;
//	}
//
//	void set_m_L(int L) {
//		m_L = L;
//	}
//
//	void set_m_W(int W) {
//		m_W = W;
//	}
//
//	int get_area() {
//		int area = m_H * m_L * 2 + m_L * m_W * 2 + m_H * m_W * 2;
//		return area;
//	}
//
//	int get_volume() {
//		int volume = m_H * m_L * m_W;
//		return volume;
//	}
//
//	bool isSameByClass(Cube& c) {
//		if (c.get_H() == get_H() && c.get_L() == get_L() && c.get_W() == get_W()) {
//			return true;
//		}
//		return false;
//	}
//};
//
//bool isSame(Cube &c1,Cube &c2){
//	if (c1.get_H() == c2.get_H() && c1.get_L() == c2.get_L() && c1.get_W() == c2.get_W()) {
//		return true;
//	}
//	return false;
//}
//
//int main6() {
//	//创建立方体对象
//	Cube c1;
//	c1.set_m_H(10);
//	c1.set_m_L(10);
//	c1.set_m_W(10);
//	Cube c2;
//	c2.set_m_H(10);
//	c2.set_m_L(5);
//	c2.set_m_W(10);
//	cout<<"立方体面积为"<<c1.get_area()<<endl;
//	cout<<"立方体体积为"<<c1.get_volume()<<endl;
//
//
//	//bool ret = isSame(c1, c2);
//	//if (ret) {
//	//	cout << "c1和c2相等" << endl;
//	//}
//	//else { cout << "c1和c2不相等" << endl; }
//
//	bool classret = c1.isSameByClass(c2);
//	if (classret) {
//		cout << "c1和c2相等" << endl;
//	}
//	else { cout << "c1和c2不相等" << endl; }
//
//	system("pause");
//	return 0;
//}