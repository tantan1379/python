#include <iostream>
#include "function.h"
using namespace std;

int main() {
	BOOKS abs;
	//初始化通讯录中当前人员的个数
	abs.m_Size = 0;
	int select = 0;
	while (true) {
		showMenu();
		cout << "please input the mode:";
		cin >> select;
		switch (select) {
		case 1://Add contact
			addContact(&abs);
			break;
		case 2://Show contact
			showContact(&abs);
			break;
		case 3://Delete contact
			deleteContact(&abs);
			break;
		case 4://Find contact
			findContact(&abs);
			break;
		case 5://Modify contact
			modifyContact(&abs);
			break;
		case 6://Clean contact
			cleanContact(&abs);
			break;
		case 0://Exit
			cout << "Thanks for using! " << endl;
			system("pause");
			return 0;
			break;
		default:
			cout << "invalid input!" << endl;
			system("pause");
			system("cls");
			break;
		}
	}

	system("pause");
	return 0;
}

void showMenu() {
	cout << "****************" << endl;
	cout << "1、Add contact" << endl;
	cout << "2、Show contact" << endl;
	cout << "3、Delete contact" << endl;
	cout << "4、Find contact" << endl;
	cout << "5、Modify contact" << endl;
	cout << "6、Clear contact" << endl;
	cout << "0、Exit" << endl;
	cout << "****************" << endl;
}

void addContact(BOOKS* abs) {
	if (abs->m_Size >= 50)
	{
		cout << "Address book has been full and can't be added." << endl;
		return;
	}
	else {
		//input name
		string name;
		cout << "Please input the contact name：" << endl;
		cin >> name;
		abs->contactArray[abs->m_Size].m_Name = name;
		//input sex
		cout << "Please input the contact sex(male/female[1/2])：" << endl;
		int sex = 0;
		while (true) {
			cin >> sex;
			if (sex == 1 || sex == 2) {
				abs->contactArray[abs->m_Size].m_Sex = sex;
				break;
			}
			cout << "Invalid input, please retype!" << endl;
			cout << "Please input the contact sex(male/female[1/2])：" << endl;
		}
		//input age
		int age = 0;
		cout << "Please input the contact age：" << endl;
		cin >> age;
		abs->contactArray[abs->m_Size].m_Age = age;

		//input phone number
		string phone;
		cout << "Please input the contact phone number：" << endl;
		cin >> phone;
		abs->contactArray[abs->m_Size].m_Phone = phone;

		//input address
		string address;
		cout << "Please input the contact address：" << endl;
		cin >> address;
		abs->contactArray[abs->m_Size].m_Address = address;

		//refresh address book number
		abs->m_Size++;
		cout << "Add contact successfully" << endl;
		system("pause");
		system("cls");
	}
}

void showContact(BOOKS* abs) {
	if (abs->m_Size == 0) {
		cout << "Address book is empty!" << endl;
	}
	else {
		for (int i = 0; i < abs->m_Size; i++) {

			cout << "name:" << abs->contactArray[i].m_Name << "\t";
			cout << "sex:" << (abs->contactArray[i].m_Sex == 1 ? "男" : "女") << "\t";
			cout << "age:" << abs->contactArray[i].m_Age << "\t";
			cout << "phone:" << abs->contactArray[i].m_Phone << "\t";
			cout << "address:" << abs->contactArray[i].m_Address << endl;
		} 
	}
	system("pause");
	system("cls");
}

//detect whether the person exists
int isExist(BOOKS* abs, string name) {
	for (int i = 0; i < abs->m_Size; i++) {
		if ((abs->contactArray[i].m_Name) == name) {
			return i;
		}
	}
	return -1;
}

void deleteContact(BOOKS* abs) {//这里的abs为结构体指针变量，相当于传入结构体变量的地址，而不再是结构体的变量名
	if (abs->m_Size == 0) {
		cout << "Address book is empty!" << endl;
	}
	else {
		string name;
		cout << "please input the contact you need to delete" << endl;
		cin >> name;
		int index = isExist(abs, name);
		if (index != -1) {
			for (int i = index; i < abs->m_Size; i++) {
				//将需要删除的第i个元素用i+1个元素进行覆盖，则相当于删除
				abs->contactArray[i] = abs->contactArray[i + 1];
			}
			cout << name << " has been deleted" << endl;
			//refresh address book number
			abs->m_Size--;
		}
		else {
			cout << "Person " << name << " not found, please retype!" << endl;
		}
	}
	system("pause");
	system("cls");
}

void findContact(BOOKS* abs) {
	if (abs->m_Size == 0) {
		cout << "Address book is empty!" << endl;
	}
	else {
		string name;
		cout << "Please input the contact you need to find:" << endl;
		cin >> name;
		int index = isExist(abs, name);
		if (index != -1) {
			cout << "name:" << abs->contactArray[index].m_Name << "\t";
			cout << "sex:" << abs->contactArray[index].m_Sex << "\t";
			cout << "age:" << abs->contactArray[index].m_Age << "\t";
			cout << "phone:" << abs->contactArray[index].m_Phone << "\t";
			cout << "address:" << abs->contactArray[index].m_Address << endl;
		}
		else {
			cout << "Person " << name << " not found, please retype!" << endl;
		}
	}
	system("pause");
	system("cls");
}

void modifyContact(BOOKS* abs) {
	if (abs->m_Size == 0) {
		cout << "Address book is empty!" << endl;
	}
	else {
		string name;
		cout << "Please input the contact you need to find:" << endl;
		cin >> name;
		int index = isExist(abs, name);
		if (index != -1) {
			//name
			string name;
			cout << "please input the contact name:" << endl;
			cin >> name;
			abs->contactArray[index].m_Name = name;

			//sex
			int sex = 0;
			cout << "please input the contact sex(male/female[1/2]):" << endl;
			while (true) {
				cin >> sex;
				if (sex == 1 || sex == 2) {
					abs->contactArray[index].m_Sex = sex;
					break;
				}
				cout << "Invalid input, please retype!" << endl;
				cout << "Please input the contact sex(male/female[1/2])：" << endl;
			}	
			//age
			int age = 0;
			cout << "please input the contact age:" << endl;
			cin >> age;
			abs->contactArray[index].m_Age = age;
			//phone
			string phone;
			cout << "please input the contact phone:" << endl;
			cin >> phone;
			abs->contactArray[index].m_Phone = phone;
			//address
			string address;
			cout << "please input the contact address:" << endl;
			cin >> address;
			abs->contactArray[index].m_Address = address;
		}
		else {
			cout << "Person " << name << " not found, please retype!" << endl;
		}
	}
	system("pause");
	system("cls");
}

void cleanContact(BOOKS* abs) {
	abs->m_Size = 0;
	cout << "The address book has been cleaned!" << endl;
	system("pause");
	system("cls");
}
