#include <iostream>
using namespace std;

int main()
{
    //char ch;
    //cout << "This program has paused. Press Enter to continue.";
    //cin.get(ch);
    //cout << "It has paused a second time. Please press Enter again."; ch = cin.get();
    //cout << "It has paused a third time. Please press Enter again.";
    //cin.get();
    //cout << "Thank you! \n";
    int a = 1;
    if (0 == --a) {
        cout << "yes" << endl;
    }
    else {
        cout << "no" << endl;
    }
    return 0;
}