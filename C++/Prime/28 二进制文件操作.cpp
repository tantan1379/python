//#include <iostream>
//#include <fstream>
//using namespace std;
//
//class Person {
//public:
//	Person() {}
//	Person(const char* name, int age) {
//		this->age = age;
//		strcpy_s(this->name, name);
//	}
//	const char* get_name() {
//		return this->name;
//	}
//	int get_age() {
//		return this->age;
//	}
//private:
//	char name[64];
//	int age;
//};
//
//void test01() {
//	ofstream ofs("./person.txt", ios::out | ios::binary);
//	Person p("张三", 18);
//	ofs.write((const char*)&p, sizeof(Person));
//	ofs.close();
//	ifstream ifs("./person.txt", ios::in | ios::binary);
//	if (!ifs) {
//		cout << "文件打开失败！" << endl;
//		return;
//	}
//	Person p1;
//	ifs.read((char*)&p1, sizeof(p1));
//	cout << "姓名：" << p1.get_name() << " 年龄：" << p1.get_age() << endl;
//	ifs.close();
//}
//
//int main() {
//	test01();
//	system("pause");
//	return 0;
//}
