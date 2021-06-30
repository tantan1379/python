#include <unordered_map>
#include <iostream>
using namespace std;

int main() {
	unordered_map<int, int> hashmap = {
		{1,10},
		{2,20},
		{3,30}
	};
	//cout << hashmap.at(1) << endl;
	hashmap.emplace(4, 40);
	hashmap.emplace(5, 50);
	cout << hashmap.size() << endl;
	for (auto iter = hashmap.begin(); iter != hashmap.end();iter++) {
		cout << iter->first << " " << iter->second << endl;
	}
	if (hashmap.count(6)) {
		cout << "yes" << endl;
	}
	else {
		cout << "no" << endl;
	}
	return 0;
}