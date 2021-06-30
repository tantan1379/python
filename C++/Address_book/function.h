#pragma once
#include <iostream>
#include <string>

using namespace std;

typedef struct contact {
	string m_Name;
	int m_Sex;
	int m_Age;
	string m_Phone;
	string m_Address;
}CONTACT;

typedef struct addressbooks {
	CONTACT contactArray[50];
	int m_Size;
}BOOKS;

void showMenu();
void addContact(BOOKS*);
void showContact(BOOKS*);
int isExist(BOOKS*, string);
void deleteContact(BOOKS*);
void findContact(BOOKS*);
void modifyContact(BOOKS*);
void cleanContact(BOOKS*);