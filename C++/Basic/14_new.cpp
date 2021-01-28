#include <iostream>
using namespace std;

//在堆中创建变量
int example1()
{
    //可以在new后面直接赋值
    int* p = new int(3);
    //也可以单独赋值
    //*p = 3;

    return *p;
}

//在堆中创建数组
int* example2()
{
    //当new一个数组时，同样用一个指针接住数组的首地址
    int* q = new int[3];//new int[3]返回堆中创建数组的首地址并用指针q接收，此时q可以看做是创建数组的数组名
    for (int i = 0; i < 3; i++)
        q[i] = i;
    return q;//返回创建并赋值的数组的首地址
}

struct student
{
    string name;
    int score;
};

//在堆中创建结构体数组
student* example3()
{
    //这里是用一个结构体指针接住结构体数组的首地址
    //对于结构体指针，个人认为目前这种赋值方法比较方便
    student* stlist = new student[3]{ {"abc", 90}, {"bac", 78}, {"ccd", 93} };

    return stlist;//返回创建并添加元素的结构体数组的首地址
}



int main()
{
    int e1 = example1();
    cout << "e1: " << e1 << endl;

    int* e2 = example2();//example2返回的是数组首地址，因此用指针e2接收，e2可以看做是该数组的数组名
    for (int i = 0; i < 3; i++)
        cout << e2[i] << " ";
    cout << endl;


    student* st1 = example3();//example返回的是结构体数组的首地址，因此用指针st1接收，st1可以看做是该结构体数组的数组名

    for (int i = 0; i < 3; i++)
        cout << st1[i].name << " " << st1[i].score << endl;



    return 0;
}