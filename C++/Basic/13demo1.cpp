#include <iostream>
#include <string>

using namespace std;

typedef struct hero {
	string name;
	int age;
	string sex;
}HERO;

void allocateSpace(HERO* h) {
	string name_h[5] = { "刘备","关羽","张飞","赵云","貂蝉" };
	int age_h[5] = { 23,22,20,21,19 };
	string sex_h[5] = { "男","男", "男", "男", "女" };
	for (int i = 0; i < 5; i++) {
		(h + i)->name = name_h[i];
		(h + i)->age = age_h[i];
		(h + i)->sex = sex_h[i];
	}
}

void printHero(HERO* h) {
	for (int i = 0; i < 5; i++) {
		cout << "英雄名：" << (h + i)->name << " 年龄：" << (h + i)->age << " 性别：" << (h + i)->sex << endl;
	}
}

void bubbleSort(HERO* h) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4 - i; j++) {
			if (((h + j)->age) > (h + j + 1)->age) {
				HERO temp = *(h + j);
				*(h + j) = *(h + j + 1);
				*(h + j + 1) = temp;
			}
		}
	}
}

int main() {
	HERO hero[5];
	allocateSpace(hero);
	cout << "排序前各英雄列表为：" << endl;
	cout << "排序前各英雄列表为：" << endl;
	printHero(hero);
	bubbleSort(hero);
	cout << "排序后各英雄列表为：" << endl;
	printHero(hero);
	system("pause");
	return 0;
}