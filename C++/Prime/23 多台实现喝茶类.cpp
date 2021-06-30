//////利用多态特性实现制作饮品案例
//#include <iostream>
//
//using namespace std;
//
//class AbstractDrink {
//public:
//	virtual void boil() = 0;
//	virtual void pourincup() = 0;
//	virtual void putsomething() = 0;
//	void makeDrink() {
//		boil();
//		pourincup();
//		putsomething();
//	}
//};
//
//class Coffee :public AbstractDrink {
//public:
//	virtual void boil() {
//		cout << "烧水" << endl;
//	}
//	virtual void pourincup() {
//		cout << "倒入陶瓷杯中" << endl;
//	}
//	virtual void putsomething() {
//		cout << "加入牛奶" << endl;
//	}
//};
//
//class Tea :public AbstractDrink {
//public:
//	virtual void boil() {
//		cout << "烧水" << endl;
//	}
//	virtual void pourincup() {
//		cout << "倒入玻璃杯中" << endl;
//	}
//	virtual void putsomething() {
//		cout << "加入枸杞" << endl;
//	}
//};
//
//void makeDrink(AbstractDrink* abs) {
//	abs->makeDrink();
//}
//
//void test1() {
//	makeDrink(new Coffee);//父类指针指向子类对象 Abstract* abs = new Coffee;
//	cout << "-----------------" << endl;
//	makeDrink(new Tea);//同上
//}
//
//int main() {
//	test1();
//	system("pause");
//	return 0;
//}