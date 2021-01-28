//利用多态特性实现制作饮品案例
#include <iostream>

using namespace std;

class AbstractDrinking {
public:
	virtual void Boil() = 0;
	virtual void Brew() = 0;
	virtual void PourIntoCap() = 0;
	virtual void PutSomething() = 0;
	void makeDrink() {
		Boil();
		Brew();
		PourIntoCap();
		PutSomething();
	}
};

class Coffee :public AbstractDrinking {
public:
	void Boil() {
		cout << "煮农夫山泉" << endl;
	}
	void Brew() {
		cout << "冲泡咖啡" << endl;
	}
	void PourIntoCap() {
		cout << "倒入杯中" << endl;
	}
	void PutSomething() {
		cout << "加入牛奶" << endl;
	}
};

class Tea :public AbstractDrinking {
public:
	virtual void Boil() {
		cout << "煮矿泉水" << endl;
	}
	virtual void Brew() {
		cout << "冲泡茶叶" << endl;
	}
	virtual void PourIntoCap() {
		cout << "倒入杯中" << endl;
	}
	virtual void PutSomething() {
		cout << "加入柠檬" << endl;
	}
};


void doWork(AbstractDrinking& abs) {
	abs.makeDrink();
}

void doWork(AbstractDrinking* abs) {//AbstractDrinking * abs = new Coffee;
	abs->makeDrink();
	if (abs != NULL) {
		delete abs;
	}

}

void test01() {
	Coffee cf;
	doWork(cf);
	doWork(new Tea);
}


int main() {
	test01();
	system("pause");
	return 0;
}