//#include <iostream>
//
//using namespace std;
//
//class Animal {
//public:
//	virtual void speak() {//虚函数
//		cout << "动物说话" << endl;
//	}
//	void run() {
//		cout << "动物跑步" << endl;
//	}
//};
//
//class Cat :public Animal {
//public:
//	void speak() {//子类重写父类的虚函数
//		cout << "小猫说话" << endl;
//	}
//};
//
//class Dog :public Animal {
//public:
//	void speak() {
//		cout << "小狗说话" << endl;
//	}
//	void run() {
//		cout << "小狗跑步" << endl;
//	}
//};
//
////地址早绑定
//void doSpeak(Animal& animal) {
//	animal.speak();
//}
//
//void test01() {
//	Cat cat;
//	doSpeak(cat) ;
//	Dog dog;
//	doSpeak(dog);
//}
//
//void test02() {
//	cout << sizeof(Animal) << endl;
//	cout << sizeof(Cat) << endl;
//}
//
//int main() {
//	//test01();
//	test02();
//	system("pause");
//	return 0;
//}