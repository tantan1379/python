#include <stdio.h>
#include <unordered_map>
#include <string>
#include <stack>
#include <iostream>

using namespace std;

bool inValid(string s) {
	int n = s.size();
	if (n % 2 == 1) {
		return false;
	}
	unordered_map<char, char> pairs = {
		{'}','{'},
		{')','('},
		{']','['}
	};
	stack<char> stk;
	for (char ch : s) {
		if(pairs.count(ch)) {
			if (stk.empty() || stk.top() != pairs.at(ch)) {
				return false;
			}
			else {
				stk.pop();
			}
		}
		else {
			stk.emplace(ch);
		}
	}
	return stk.empty();
}

void test020() {
	string s1 = "({}[])";
	string s2 = "({[])}";
	cout << inValid(s1) << endl;
	cout << inValid(s2) << endl;
}