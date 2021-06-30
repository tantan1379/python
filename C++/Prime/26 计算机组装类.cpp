//#include <iostream>
//#include <string>
//using namespace std;
//
//class CPU {//CPU抽象类
//public:
//	virtual void caculate() = 0;
//};
//
//class GPU {//GPU抽象类
//public:
//	virtual void display() = 0;
//};
//
//class Memory {//Memory抽象类
//public:
//	virtual void storage() = 0;
//};
//
//class Computer {
//public:
//	Computer(CPU* c, GPU* g, Memory* m) {
//		cpu = c;
//		gpu = g;
//		memory = m;
//	}
//
//	void work() {
//		cpu->caculate();
//		gpu->display();
//		memory->storage();
//	}
//	~Computer() {
//		if (cpu != NULL) {
//			delete cpu;
//			cpu = NULL;
//		}		
//		if (gpu != NULL) {
//			delete gpu;
//			gpu = NULL;
//		}		
//		if (memory != NULL) {
//			delete memory;
//			memory = NULL;
//		}
//	}
//private:
//	CPU* cpu;
//	GPU* gpu;
//	Memory* memory;
//};
//
//class IntelCPU :public CPU {
//public:
//	virtual void caculate() {
//		cout << "Intel的CPU工作" << endl;
//	}
//};
//
//class IntelGPU :public GPU {
//public:
//	virtual void display() {
//		cout << "Intel的GPU显示" << endl;
//	}
//};
//
//class IntelMemory :public Memory {
//public:
//	virtual void storage() {
//		cout << "Intel的内存条存储" << endl;
//	}
//};
//
//class LenovoCPU :public CPU {
//public:
//	virtual void caculate() {
//		cout << "Lenovo的CPU工作" << endl;
//	}
//};
//
//class LenovoGPU :public GPU {
//public:
//	virtual void display() {
//		cout << "Lenovo的GPU显示" << endl;
//	}
//};
//
//class LenovoMemory :public Memory {
//public:
//	virtual void storage() {
//		cout << "Lenovo的内存条存储" << endl;
//	}
//};
//
//void test01() {
//	CPU* intelCpu = new IntelCPU;
//	GPU* intelGpu = new IntelGPU;
//	Memory* intelMemory = new IntelMemory;
//	//组装地一台电脑
//	cout << "第一台电脑开始工作！" << endl;
//	Computer* computer1 = new Computer(intelCpu, intelGpu, intelMemory);
//	computer1->work();
//	delete computer1;	
//	cout << "---------------" << endl;
//	cout << "第二台电脑开始工作！" << endl;
//	Computer* computer2 = new Computer(new LenovoCPU, new LenovoGPU, new LenovoMemory);
//	computer2->work();
//	delete computer2;
//	cout << "---------------" << endl;
//	cout << "第三台电脑开始工作！" << endl;
//	Computer* computer3 = new Computer(new IntelCPU, new LenovoGPU, new IntelMemory);
//	computer3->work();
//	delete computer3;
//}
//
//int main() {
//	test01();
//	system("pause");
//	return 0;
//}