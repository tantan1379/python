#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void test01() {
	fstream ofs;
	//写数据
	ofs.open("./test.txt");
	ofs << "I am Iron Man" << endl;
	ofs.close();
	//读数据
	ifstream ifs;
	ifs.open("./test.txt", ios::in);
	if (!ifs) {
		cout << "文件打开失败！" << endl;
		return;
	}
	//1、方法一
	char buf[1024] = { 0 };//所有元素赋值为0
	while (ifs >> buf) {
		cout << buf << endl;
	}
	//2、方法二
	//char buf[1024] = { 0 };//所有元素赋值为0
	//while (ifs.getline(buf,sizeof(buf),' ')) {
	//	cout << buf << endl;
	//}
	//3、方法三
	//string buf;
	//while (getline(ifs, buf,' ')) {
	//	cout << buf << endl;
	//}
	//4、方法四
	//char c;
	//while ((c = ifs.get()) != EOF) {
	//	cout << c;
	//}

	ifs.close();
}

int main() {
	test01();
	system("pause");
	return 0;
}