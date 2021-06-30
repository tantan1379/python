//#include "circle.h"
//#include <iostream>
//using namespace std;
//
//void Circle::set_r(int r) {
//	c_r = r;
//}
//void Circle::set_center(Point center) {
//	c_center = center;
//}
//int Circle::get_r() {
//	return c_r;
//}
//Point Circle::get_center() {
//	return c_center;
//}
//void Circle::point_circle(Point point) {
//	int distance_2 = (c_center.get_y() - point.get_y()) * (c_center.get_y() - point.get_y()) +
//		(c_center.get_x() - point.get_x()) * (c_center.get_x() - point.get_x());
//	if (distance_2 > c_r * c_r) {
//		cout << "点在圆外" << endl;
//	}
//	else if (distance_2 == c_r * c_r) {
//		cout << "点在圆上" << endl;
//	}
//	else cout << "点在圆内" << endl;
//}