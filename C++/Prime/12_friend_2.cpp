////类做友元
//#include <iostream>
//#include <string>
//
//using namespace std;
//
//class Building;
//
//class GoodGay {
//public:
//	GoodGay();
//	void visit();
//	Building* building;
//};
//
//class Building {
//	//GoodGay是本类的友类，可以访问Building类的私有内容
//	friend class GoodGay;
//public:
//	Building();
//	string m_SittingRoom;
//private:
//	string m_BedRoom;
//};
//
////类外实现构造函数
//Building::Building() {
//	m_SittingRoom = "客厅";
//	m_BedRoom = "卧室";
//}
//
////类外实现构造函数
//GoodGay::GoodGay() {
//	building = new Building;
//}
//
////类外实现成员函数
//void GoodGay::visit() {
//	cout << "好基友正在访问：" << this->building->m_SittingRoom << endl;
//	cout << "好基友正在访问：" << this->building->m_BedRoom << endl;
//}
//
//void test01() {
//	GoodGay gg;
//	gg.visit();
//}
//
//int main() {
//	test01();
//
//	system("pause");
//	return 0;
//}