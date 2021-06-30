#pragma once
#include <iostream>
#include <string>
#include <list>
#include <fstream>
#include "Administrator.h"
#include "Understudent.h"

using namespace std;

class System {
private:
	list<Understudent>underst;
	list<Administrator>ad;
	static int underst_count;
	static int ad_count;
public:
	//外部接口
	//基本功能
	virtual void system_interface();			//登录界面
	void system_functionshow();					//系统功能显示
	void underst_functionshow(string name);		//本科生系统显示界面
	void ad_functionshow(string name);			//管理员系统显示界面
	void save_underst();						//保存本科生数据库
	void save_ad();								//保存管理员数据库
	void load_underst();						//加载本科生数据库
	void load_ad();								//加载管理员数据库
	//系统功能
	void set_ad_account();						//(1)设置管理员账号
	void enter_ad_account();					//(2)管理员登录
	void enter_underst_account();				//(3)本科生登录
	void look_all_ad();							//(4)查看管理员信息
	void clear_all_ad();						//(5)删除所有管理员信息
	void exit_system();							//(6)退出系统

	//内部接口
	//管理员功能
	void look_all_underst();					//(1)查看所有本科生数据
	void look_all_underst_by_grade();			//(2)按成绩排列查看所有本科生数据
	void look_underst_by_name();				//(3)按名称查询本科生信息
	void look_underst_by_id();					//(4)按学号查询本科生信息
	void input_underst_info();					//(5)录入本科生信息
	void delete_underst_by_id();				//(6)按学号删除本科生信息
	void delete_underst_all();					//(7)清空所有本科生信息
	void change_ad_password(string id);			//(8)修改管理员密码
	//本科生功能
	void change_underst_password(string id);	//(1)修改本科生密码
};
