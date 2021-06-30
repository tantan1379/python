//#include <iostream>
//#include <string>
//using namespace std;
//
//class Animal {
//public:
//	Animal() {
//		cout << "Animal构造函数调用" << endl;
//	}
//	//virtual ~Animal() {
//	//	cout << "Animal析构函数调用" << endl;
//	//}
//	virtual ~Animal() = 0;
//	virtual void speak() = 0;
//};
//Animal::~Animal() {
//	cout << "Animal的纯虚析构函数" << endl;
//}
//
//class Cat :public Animal {
//public:
//	Cat(string n) {
//		cout << "Cat构造函数调用" << endl;
//		name = new string(n);
//	}
//	~Cat() {
//		if (name != NULL) {
//			cout << "Cat析构函数调用" << endl;
//			delete name;
//			name = NULL;
//		}
//	}
//	virtual void speak() {
//		cout << *name << "小猫在说话" << endl;
//	}
//
//	string* name;
//
//};
//
//void test01() {
//	Animal* animal = new Cat("Tom");//父类指针指向子类对象
//	animal->speak();
//	delete animal;//父类指针析构时不会调用子类的析构函数，导致子类会出现内存泄漏
//}
//
//int main() {
//	test01();
//	system("pause");
//	return 0;
//}