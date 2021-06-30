//#include <iostream>
//
//using namespace std;
//
////实现计算器抽象类
//class AbstractCaculator {
//public:
//	int m_Num1;
//	int m_Num2;  
//	virtual int getResult(){};
//};
//
////加法计算器类
//class AddCaculator :public AbstractCaculator {
//public:
//	int getResult() {
//		return m_Num1 + m_Num2;
//	}
//};
//
////减法计算器类
//class SubCaculator :public AbstractCaculator {
//public:
//	int getResult() {
//		return m_Num1 - m_Num2;
//	}
//};
//
////乘法计算器类
//class MulCaculator :public AbstractCaculator {
//public:
//	int getResult() {
//		return m_Num1 * m_Num2;
//	}
//};
//
//void test01() {
//	AbstractCaculator* abc = new AddCaculator;//父类的指针指向子类对象
//	abc->m_Num1 = 10;
//	abc->m_Num2 = 10;
//	cout << abc->m_Num1 << " + " << abc->m_Num2 << " = " << abc->getResult() << endl;
//	delete abc;
//
//	abc = new SubCaculator;
//	abc->m_Num1 = 10;
//	abc->m_Num2 = 10;
//	cout << abc->m_Num1 << " - " << abc->m_Num2 << " = " << abc->getResult() << endl;
//	delete abc;
//}
//
//int main() {
//
//	test01();
//
//	system("pause");
//	return 0;
//}