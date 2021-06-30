//#include <iostream>
//using namespace std;
//class Date;
//class Time {
//private:
//	int hour, minute, second;
//public:
//	void display(Date&);
//	Time(int h, int m, int s) :hour(h), minute(m), second(s) {}
//};
//
//class Date {
//private:
//	int year, month, day;
//public:
//	Date(int h, int m, int s) :year(h), month(m), day(s) {}
//	friend void Time::display(Date&);//Time的公共函数display可以使用Date的私有变量
//};
//
//void Time::display(Date& d) {
//	cout << d.year << ":" << d.month << ":" << d.day << endl;
//	cout << hour << ":" << minute << ":" << second << endl;
//}
//
//void test01() {
//	Time t1(12, 30, 11);
//	Date d1(2021, 5, 17);
//	t1.display(d1);
//}
//
//int main()
//{
//	test01();
//	system("pause");
//	return 0;
//}