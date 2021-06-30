#include <iostream>
#include <string>
#include <time.h>

using namespace std;

typedef struct student {
	string name;
	int score;
}STUDENT;

typedef struct teacher {
	string name;
	STUDENT stuarray[5];
}TEACHER;

void allocateSpace(TEACHER* t) {
	string tname[3] = { "chen","zhu","nie" };
	string sname[5] = { "tanwenhao","liudengfeng","gaosong","shenjiayan","liuming" };
	for (int i = 0; i < 3; i++) {
		(t + i)->name = tname[i];
		for (int j = 0; j < 5; j++) {
			(t + i)->stuarray[j].name = sname[j];
			(t + i)->stuarray[j].score = rand() % 60 + 40;
		}
	}
}

void print_teacher(TEACHER* t) {
	for (int i = 0; i < 3; i++) {
		cout << "老师姓名：" << (t + i)->name << endl;
		for (int j = 0; j < 5; j++) {
			cout << "\t姓名：" << (t + i)->stuarray[j].name << "  分数：" << (t + i)->stuarray[j].score << endl;
		}
	}
}

int main12() {
	srand((unsigned int)time(NULL));
	TEACHER teaarray[3];
	allocateSpace(teaarray);
	print_teacher(teaarray);
	system("pause");
	return 0;
}