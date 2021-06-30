#include "System.h"
int System::underst_count = 0;
int System::ad_count = 0;

//系统功能显示
void System::system_functionshow() {
	cout << "--------------------------------------------" << endl;
	cout << endl;
	cout << "欢迎使用学生管理系统！" << endl;
	cout << "*********************" << endl;
	cout << "(1)管理员登录" << endl;
	cout << "(2)学生登录" << endl;
	cout << "(3)设置管理员" << endl;
	cout << "(4)查看管理员信息" << endl;
	cout << "(5)删除所有管理员信息" << endl;
	cout << "(6)退出系统" << endl;
	cout << "*********************" << endl;
	cout << "请输入您的操作(1-6):";
}

//管理员系统显示界面
void System::ad_functionshow(string name) {
	cout << "***管理员：" << name << " 您好！***" << endl;
	cout << "************************" << endl;
	cout << "(1)查看所有本科生数据" << endl;
	cout << "(2)按成绩排列查看" << endl;
	cout << "(3)按名称查询本科生信息" << endl;
	cout << "(4)按学号查询本科生信息" << endl;
	cout << "(5)录入本科生信息" << endl;
	cout << "(6)按学号删除本科生信息" << endl;
	cout << "(7)清空所有本科生信息" << endl;
	cout << "(8)修改密码" << endl;
	cout << "(9)注销账号" << endl;
	cout << "************************" << endl;
	cout << "请输入您的操作(1-9):";
}

//本科生系统显示界面
void System::underst_functionshow(string name) {
	cout << name << "同学，你好！";
	cout << "******************" << endl;
	cout << "(1)查看个人信息" << endl;
	cout << "(2)修改密码" << endl;
	cout << "(3)注销账号" << endl;
	cout << "******************" << endl;
	cout << "请输入您的操作(1-3):";
}

//登录界面
void System::system_interface() {
	while (1) {
		system("cls");
		load_underst();
		load_ad();
		system_functionshow();
		int s;
		cin >> s;
		while (s < 1 || s>6) {
			cout << "非法操作，请重新输入(1-6):";
			cin >> s;
		}
		switch (s) {
		case 1:
			enter_ad_account();
			break;
		case 2:
			enter_underst_account();
			break;
		case 3:
			set_ad_account();
			break;
		case 4:
			look_all_ad();
			break;
		case 5:
			clear_all_ad();
			break;
		case 6:
			exit_system();
			break;
		default:
			break;
		}
	}
}

//保存本科生数据库
void System::save_underst() {
	ofstream outfile("understudent.dat", ios::out);
	list<Understudent>::iterator iter;
	outfile << underst_count << endl;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		outfile << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_password()
			<< "\t" << iter->get_sex() << "\t" << iter->get_grade() << endl;
	}
	outfile.close();
}

//保存管理员数据库
void System::save_ad() {
	ofstream outfile("administrator.dat", ios::out);
	list<Administrator>::iterator iter;
	outfile << ad_count << endl;
	for (iter = ad.begin(); iter != ad.end(); iter++) {
		outfile << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_password() << endl;
	}
	outfile.close();
}

//加载本科生数据库
void System::load_underst() {
	ifstream infile("understudent.dat", ios::in);
	if (!infile) {
		cout << "注意：无本科生数据！" << endl;
		return;
	}
	string name;
	string id;
	string sex;
	string password;
	float grade;
	if (!underst.empty()) {
		underst.clear();
	}
	infile >> underst_count;
	infile.get();
	while (!infile.eof() && infile.peek() != EOF) {
		infile >> name >> id >> password >> sex >> grade;
		infile.get();
		Understudent undst(name, id, password, sex, grade);
		underst.push_back(undst);
	}
	infile.close();
	cout << "本科生数据已读取！" << endl;
}

//加载管理员数据库
void System::load_ad() {
	ifstream infile("administrator.dat", ios::in);
	if (!infile) {
		cout << "注意：读取失败，请检查数据文件是否存在！" << endl;
		return;
	}
	string name;
	string id;
	string password;
	if (!ad.empty()) {
		ad.clear();
	}
	infile >> ad_count;
	infile.get();
	if (ad_count == 0) {
		cout << "注意：尚未注册管理员，请先注册管理员再进行其他操作！" << endl;
		return;
	}
	while (!infile.eof() && infile.peek() != EOF) {
		infile >> name >> id >> password;
		infile.get();
		Administrator adm(name, id, password);
		ad.push_back(adm);
	}
	infile.close();
	cout << "管理员数据已读取！" << endl;
}

//(1)管理员登录
void System::enter_ad_account() {
	if (ad.empty() || ad_count == 0) {
		cout << endl<<"您尚未设置管理员，请先注册管理员！" << endl;
		cout << endl;
		system("pause");
		return;
	}
	string id;
	string pass;
	int flag = 1;
	int n;
	list<Administrator>::iterator iter;
	cout << endl << "请输入管理员工号：";
	cin >> id;
	for (iter = ad.begin(); iter != ad.end(); iter++) {
		if (iter->get_id() == id) {
			flag = 0;
			break;
		}
	}
	if (flag) {
		cout << endl << "该账户不存在！" << endl;
		cout << endl;
		system("pause");
		return;
	}
	cout << endl << "请输入密码：";
	cin >> pass;
	while (iter->get_password() != pass) {
		cout << endl << "密码错误！请重新输入：";
		cin >> pass;
	}
	cin.get();
	while (1) {
		system("cls");
		ad_functionshow(iter->get_name());
		cin >> n;
		while (n < 1 || n>9) {
			cout << "请输入正确的选项(1-9)：";
			cin >> n;
		}
		switch (n)
		{
		case 1:
			look_all_underst();
			break;
		case 2:
			look_all_underst_by_grade();
			break;
		case 3:
			look_underst_by_name();
			break;
		case 4:
			look_underst_by_id();
			break;
		case 5:
			input_underst_info();
			break;
		case 6:
			delete_underst_by_id();
			break;
		case 7:
			delete_underst_all();
			break;
		case 8:
			change_ad_password(iter->get_id());
			break;
		case 9:
			return;
			break;
		default:
			break;
		}
	}
}

//(2)本科生登录
void System::enter_underst_account() {
	string id;
	string pass;
	int flag = 1;
	int n;
	list<Understudent>::iterator iter;
	cout << endl << "请输入学号：";
	cin >> id;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		if (iter->get_id() == id) {
			flag = 0;
			break;
		}
	}
	if (flag) {
		cout << endl << "该学号不存在！" << endl;
		cout << endl;
		system("pause");
		return;
	}
	cout << endl << "请输入密码：";
	cin >> pass;
	while (iter->get_password() != pass) {
		cout << endl << "密码错误！请重新输入：";
		cin >> pass;
	}
	cin.get();
	while (1) {
		system("cls");
		underst_functionshow(iter->get_name());
		cin >> n;
		switch (n)
		{
		case 1:
			system("cls");
			iter->display();
			system("pause");
			break;
		case 2:
			change_underst_password(iter->get_id());
			break;
		case 3:
			return;
			break;
		default:
			break;
		}
	}
}

//(3)设置管理员账号
void System::set_ad_account() {
	list<Administrator>::iterator iter;
	string name;
	string pass;
	string id;
	cout << endl << "请输入管理员姓名：";
	cin >> name;
	cout << endl << "请输入管理员工号：";
	cin >> id;
	for (iter = ad.begin(); iter != ad.end(); iter++) {
		if (iter->get_id() == id) {
			cout << endl << "该工号已被注册！" << endl;
			cout << endl;
			system("pause");
			return;
		}
	}
	cout << endl << "请输入密码：";
	cin >> pass;
	Administrator adm(name, pass, id);
	ad.push_back(adm);
	ad_count++;
	cout << endl << "账户创建成功！";
	system("pause");
	save_ad();
}

//(4)查看管理员信息
void System::look_all_ad() {
	system("cls");
	list<Administrator>::iterator iter;
	cout << "姓名" << "\t" << "工号" << "\t" << "密码" << endl;
	for (iter = ad.begin(); iter != ad.end(); iter++) {
		cout << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_password() << endl;
	}
	cout << endl << "管理员总数为：" << ad_count << endl;
	cout << endl;
	system("pause");
}

//(5)删除所有管理员信息
void System::clear_all_ad() {
	char s;
	cout << endl << "是否确定删除所有管理员信息（一经删除无法恢复！）(Y/N):";
	cin >> s;
	while (s != 'y' && s != 'Y' && s != 'n' && s != 'N') {
		cout << endl << "请输入合法操作！(Y/N):";
		cin >> s;
	}
	if (s == 'y' || s == 'Y') {
		ad.clear();
		ad_count = 0;
		save_ad();
		cout << endl << "管理员数据已删除！请重新注册新的管理员！";
		system("pause");
	}
}

//(6)退出系统
void System::exit_system() {
	cout << endl << "感谢使用！" << endl;
	system("pause");
	exit(0);
}

//(1)查看所有本科生数据
void System::look_all_underst() {
	system("cls");
	if (underst.empty()) {
		cout << "无本科生数据!" << endl;
		system("pause");
		return;
	}
	list<Understudent>::iterator iter;
	cout << "姓名" << "\t" << "学号" << "\t" << "性别" << "\t" << "绩点" << endl;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		cout << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_sex() << "\t" << iter->get_grade() << endl;
	}
	cout << endl << "-----------------------------" << endl;
	cout << "本科生总数为：" << underst_count << endl;
	cout << endl;
	system("pause");
}

//(2)按成绩排列查看所有本科生数据
void System::look_all_underst_by_grade() {
	system("cls");
	if (underst.empty()) {
		cout << "无本科生数据!" << endl;
		system("pause");
		return;
	}
	list<Understudent>underst_copy;
	list<Understudent>::iterator iter;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		underst_copy.push_back(*iter);
	}
	underst_copy.sort();
	cout << "姓名" << "\t" << "学号" << "\t" << "性别" << "\t" << "绩点" << endl;
	for (iter = underst_copy.begin(); iter != underst_copy.end(); iter++) {
		cout << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_sex() << "\t" << iter->get_grade() << endl;
	}
	cout << endl << "-----------------------------" << endl;
	cout << "本科生总数为：" << underst_count << endl;
	cout << endl;
	system("pause");
}

//(3)按名称查询本科生信息
void System::look_underst_by_name() {
	int flag = 1;
	if (underst.empty()) {
		cout << "无本科生数据!";
		system("pause");
		return;
	}
	system("cls");
	string name_query;
	cout << "请输入需要查询学生的姓名：";
	cin >> name_query;
	list<Understudent>::iterator iter;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		if (iter->get_name() == name_query) {
			if (flag == 1) {
				cout << endl << "-----------------------------" << endl;
				cout << "姓名" << "\t" << "学号" << "\t" << "性别" << "\t" << "绩点" << endl;
				flag = 0;
			}
			cout << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_sex() << "\t" << iter->get_grade() << endl;
			system("pause");
			return;
		}
	}
	cout << "该学生不存在，请重试！" << endl;
	cout << endl;
	system("pause");
	return;
}

//(4)按学号查询本科生信息
void System::look_underst_by_id() {
	if (underst.empty()) {
		cout << endl<<"无本科生数据!" << endl;
		system("pause");
		return;
	}
	system("cls");
	int flag = 1;
	string id_query;
	cout << "请输入需要查询学生的学号：";
	cin >> id_query;
	list<Understudent>::iterator iter;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		if (iter->get_id() == id_query) {
			if (flag == 1) {
				cout << endl << "-----------------------------" << endl;
				cout << "姓名" << "\t" << "学号" << "\t" << "性别" << "\t" << "绩点" << endl;
				flag = 0;
			}
			cout << iter->get_name() << "\t" << iter->get_id() << "\t" << iter->get_sex() << "\t" << iter->get_grade() << endl;
			system("pause");
			return;
		}
	}
	cout << "该学生不存在，请重试！" << endl;
	cout << endl;
	system("pause");
	return;
}

//(5)录入本科生信息
void System::input_underst_info() {
	string name;
	string pass;
	string sex;
	string id;
	float grade;
	char s = 'Y';
	while (s == 'Y' || s == 'y') {
		system("cls");
		cout << endl << "请输入学生姓名：";
		cin >> name;
		cout << endl << "请输入学生学号：";
		cin >> id;
		cout << endl << "请输入学生初始密码：";
		cin >> pass;
		cout << endl << "请输入学生性别：";
		cin >> sex;
		cout << endl << "请输入学生绩点：";
		cin >> grade;
		Understudent undst(name, id, pass, sex, grade);
		underst.push_back(undst);
		underst_count++;
		cout << endl << "是否继续录入(Y/N)：";
		save_underst();
		cin >> s;
		while (s != 'y' && s != 'Y' && s != 'n' && s != 'N') {
			cout << endl << "请输入合法操作！(Y/N):";
			cin >> s;
		}
	}
	save_underst();
}

//(6)按学号删除本科生信息
void System::delete_underst_by_id() {
	if (underst.empty()) {
		cout << "无本科生数据!" << endl;
		system("pause");
		return;
	}
	string name;
	string sex;
	string password;
	float grade;
	string delete_id;
	cout << endl << "请输入需要删除的学生信息的学号：";
	cin >> delete_id;
	list<Understudent>::iterator iter;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		if (delete_id == iter->get_id()) {
			name = iter->get_name();
			sex = iter->get_sex();
			password = iter->get_password();
			grade = iter->get_grade();
			Understudent undst(name, delete_id, password, sex, grade);
			underst.remove(undst);
			underst_count--;
			if (underst_count < 0) {
				cout << "未知错误！" << endl;
			}
			save_underst();
			cout << endl<<"已删除学生" << name << "!";
			system("pause");
			return;
		}
	}
}

//(7)清空所有本科生信息
void System::delete_underst_all() {
	char s;
	cout << endl << "是否确定删除所有本科生信息（一经删除无法恢复！）(Y/N):";
	cin >> s;
	while (s != 'y' && s != 'Y' && s != 'n' && s != 'N') {
		cout << endl << "请输入合法操作！(Y/N):";
		cin >> s;
	}
	if (s == 'y' || s == 'Y') {
		underst.clear();
		underst_count = 0;
		save_ad();
		cout << endl << "本科生数据已删除！";
		system("pause");
	}
}

//(8)修改管理员密码
void System::change_ad_password(string id) {
	list<Administrator>::iterator iter;
	for (iter = ad.begin(); iter != ad.end(); iter++) {
		if (iter->get_id() == id) {
			break;
		}
	}
	string pass;
	string pass_new;
	cout << endl<<"请输入原密码：";
	cin >> pass;
	if (pass !=iter->get_password()) {
		cout << endl << "密码错误！" << endl;
		system("pause");
		return;
	}
	cout <<endl<<"请输入新密码：";
	cin >> pass_new;
	iter->set_password(pass_new);
	cout <<endl<< "密码修改成功！";
	save_ad();
	system("pause");

}

//(1)修改本科生密码
void System::change_underst_password(string id) {
	list<Understudent>::iterator iter;
	for (iter = underst.begin(); iter != underst.end(); iter++) {
		if (iter->get_id() == id) {
			break;
		}
	}
	string pass;
	string pass_new;
	cout << endl<<"请输入原密码：";
	cin >> pass;
	if (pass !=iter->get_password()) {
		cout << endl << "密码错误！" << endl;
		system("pause");
		return;
	}
	cout <<endl<<"请输入新密码：";
	cin >> pass_new;
	iter->set_password(pass_new);
	cout <<endl<< "密码修改成功！";
	save_underst();
	system("pause");
}