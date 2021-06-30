#pragma once
#include <iostream>
#include <string>

using namespace std;
class Administrator {
private:
	string name;
	string password;
	string ID;
public:
	Administrator(string n, string p, string i) :name(n), password(p), ID(i) {}
	string get_name() {
		return name;
	}
	string get_password() {
		return password;
	}
	string get_id() {
		return ID;
	}
	void display() {
		cout << "******************" << endl;
		cout << "* ÐÕÃû£º" << name << endl;
		cout << "* ¹¤ºÅ£º" << ID << endl;
		cout << "******************" << endl;
	}
	void set_password(string p) {
		password = p;
	}
};