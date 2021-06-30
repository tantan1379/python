#pragma once
#include <iostream>
#include <string>

using namespace std;

class Understudent {
private:
	string name;
	string ID;
	string password;
	string sex;
	float grade;
public:
	Understudent(string n, string i, string p, string s, float g) :name(n), ID(i), password(p), sex(s), grade(g) {}
	
	string get_name() {
		return name;
	}

	string get_id() {
		return ID;
	}

	string get_password() {
		return password;
	}

	string get_sex() {
		return sex;
	}
	
	float get_grade() {
		return grade;
	}

	void display() {
		cout << "******************" << endl;
		cout << "* 姓名：" << name << endl;
		cout << "* 学号：" << ID << endl;
		cout << "* 性别：" << sex << endl;
		cout << "* 绩点：" << grade << endl;
		cout << "******************" << endl;
	}

	void set_password(string p) {
		password = p;
	}

	bool operator == (const Understudent& u)const{
		return ID == u.ID;
	}

	bool operator <(const Understudent& u)const {
		if (grade != u.grade) { return grade < u.grade; }
		else if (name != u.name) { return name < u.name; }
		else if (ID != u.ID) { return ID < u.ID; }
		else { cout << "error comparison!" << endl; }
	}
};