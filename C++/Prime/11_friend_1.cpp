////全局函数做友元
//#include <iostream>
//#include <string>
//
//using namespace std;
//
//class Building {
//	//全局函数做友元
//	friend void goodguy(Building& building);
//public:
//	string m_SittingRoom;
//
//	Building() { 
//		m_SittingRoom = "客厅";
//		m_BedRoom = "卧室";
//	}
//
//private:
//	string m_BedRoom;
//
//};
//
////全局函数
//void goodguy(Building &building) {
//	cout << "Your friend is visiting:" << building.m_SittingRoom << endl;
//	cout << "Your friend is visiting:" << building.m_BedRoom << endl;
//
//}
//
//void test1() {
//	Building b;
//	goodguy(b);
//}
//
//int main() {
//	test1();
//
//
//	system("pause");
//	return 0;
//}