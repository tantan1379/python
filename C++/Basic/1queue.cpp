#include <stack>
#include <queue>
#include <iostream>

using namespace std;
static int a;

int main1() {
    std::queue<int> Q;
    if (Q.empty()) {
        cout << "queue is empty" << endl;
        Q.push(3);   // 最先push的位于队列的front部分
            Q.push(5);
        Q.push(10);  // 最后push的位于队列的back部分
            cout << Q.front() << endl;  // 输出3
            cout << Q.back() << endl;  // 输出10
            Q.pop();  // 弹出front部分的数值，先进先出！！！
            cout << Q.front() << endl;  // 输出5
    }
    system("pause");
    return 0;
}
