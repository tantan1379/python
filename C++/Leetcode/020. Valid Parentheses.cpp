#include <stdio.h>
#include <unordered_map>
#include <string>
#include <stack>


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
		
	}
}