//#include <iostream>
//#include <stdlib.h>
//#include <ctime>
//using namespace std;
//
////const int arr_num = 10;
////int main() {
////	int arr[arr_num];
////	int temp = 0;
////	bool flag;
////	//设置随机种子
////	srand((unsigned int)time(NULL));
////	cout << "原数组为：" << endl;
////	for (int i = 0;i < arr_num;i++){
////		//设置随机数生成范围为0-100
////		arr[i] = rand()%100;
////		cout << arr[i];
////		if (i == arr_num-1)
////			continue;
////		cout << ',';
////	}
////	cout << endl;
////	//冒泡排序
////	for (int i = 0;i < arr_num-1;i++) {
////		flag = 0;
////		for (int j = 0;j < arr_num-i-1;j++) {
////			if (arr[j] > arr[j + 1]) {
////				flag = 1;
////				temp = arr[j];
////				arr[j] = arr[j + 1];
////				arr[j + 1] = temp;
////			}
////		}
////		if (flag == 0) break;
////	}
////	//显示排序后数组
////	cout << "排序后数组为：" << endl;
////	for (int i = 0;i < arr_num;i++) {
////		cout << arr[i];
////		if (i == arr_num-1)
////			continue;
////		cout << ',';
////	}
////	cout << endl;
////
////
////
////	system("pause");
////	return 0;
////}
////int createarray(int);
//int* bubbleSort(int*, int);
//void printArray(int*, int);
//
//
//
//int main2() {
//	const int len = 10;
//	int arr[len];
//	srand((unsigned int)time(NULL));
//	cout << "原数组为：" << endl;
//	for (int i = 0;i < len;i++){
//		//设置随机数生成范围为0-100
//		arr[i] = rand()%100;
//	}
//	printArray(arr, len);//arr是数组名，用于表示arr首元素的地址，相当于&arr也相当于&arr[0]
//	bubbleSort(arr, len);
//	cout << "排序后数组为：" << endl;
//	printArray(arr, len);
//	system("pause");
//	return 0;
//}
//
////int createarray(int len) {
////	int arr[len];
////	srand((unsigned int)time(NULL));
////	for(int i=0;i<len;i++){
////		arr[i] = rand() % 100;
////	}
////	return arr;
////}
//
//void printArray(int* arr, int len) {
//	for (int i = 0;i < len;i++) {
//		cout << arr[i];
//		if (i == len - 1)
//			continue;
//		cout << ',';
//	}
//	cout << endl;
//}
//
//
//int* bubbleSort(int *arr, int len) { //  形参中的arr是指向主函数中实参arr首地址的指针 相当于int* p= &arr(写p是为了区分，实际上就是形参的arr)
//	bool flag = 0;
//	for (int i = 0;i < len - 1;i++) {
//		for (int j = 0;j < len - i - 1;j++) {
//			if (arr[j] > arr[j + 1]) {
//				int temp = arr[j + 1];
//				arr[j + 1] = arr[j];
//				arr[j] = temp;
//				flag = 1;
//			}
//		}
//		if (flag == 0)
//			return arr;
//	}
//	return arr;
//}
