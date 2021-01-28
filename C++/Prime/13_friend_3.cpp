////成员函数作友元
//#include <string>
//#include <iostream>
//
//using namespace std;
//
////Building类的声明，防止Guy中报错
//class Building;
//
//class Guy {
//public:
//	Guy();
//	~Guy();
//	void test();
//	Building* building;
//};
//
////Building类的定义
//class Building {
//	friend void Guy::test();
//public:
//	Building();
//	string m_SittingRoom;
//private:
//	string m_BedRoom;
//};
//
////类外实现Guy的构造函数，此处building被称为类指针，当new时我们为其分配内存
//Guy::Guy() {
//	building = new Building;
//}
//
////类外实现Guy的析构函数
//Guy::~Guy() {
//	if (NULL!=building) {
//		delete building;
//		building = NULL;
//	}
//}
//
////类外实现Guy的成员函数，且为Building的友元函数，可以访问Building中的私有成员
//void Guy::test() {
//	cout << "m_SittingRoom:" << building->m_SittingRoom << endl;
//	cout << "m_BedRoom:" << building->m_BedRoom << endl;
//}
//
////类外实现Building的构造函数
//Building::Building() {
//	this->m_BedRoom = "卧室";
//	this->m_SittingRoom = "客厅";
//}
//
//
//int main() {
//	Guy g;
//	g.test();
//	system("pause");
//	return 0;
//}