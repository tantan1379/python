#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2020/9/20 16:17


print('-' * 20, '欢迎来到学生管理系统', '-' * 20)
stus = ['\ttom \t18  \t男  \t北大街', '\tlucy \t19  \t女  \t胜利桥']

while True:
    print('请输入你需要完成的操作:')
    print('\t1.查询学生')
    print('\t2.添加学生')
    print('\t3.删除学生')
    print('\t4.退出系统')
    user_choose = int(input('请选择1-4:'))
    print('-' * 62)

    if user_choose == 1:
        print('\t序号 \t姓名 \t年龄 \t性别 \t地址')
        n = 1
        for stu in stus:
            print(f'\t{n}\t{stu}')
            n += 1

    elif user_choose == 2:
        stu_name = input('请输入添加学生的姓名: ')
        stu_age = input('请输入添加学生的年龄: ')
        stu_gender = input('请输入添加学生的性别: ')
        stu_add = input('请输入添加学生的地址: ')
        stu = f'\t{stu_name} \t{stu_age}  \t{stu_gender}  \t{stu_add}'
        stus.append(stu)
        print('该学生已被添加到系统中')
        print('-' * 62)
        print('\t姓名 \t年龄 \t性别 \t地址')
        print(stu)

    elif user_choose == 3:
        del_num = int(input('请输入要删除的学生的序号：'))
        if 0 < del_num <= len(stus):
            del_i = del_num - 1
            print('以下学生将被删除')
            print('-' * 62)
            print('\t姓名 \t年龄 \t性别 \t地址')
            print(stus[del_i])
            user_confirm = input('该操作不可还原，是否确定删除%d号同学的个人信息?[Y/N]' % del_num)
            if user_confirm == 'Y' or user_confirm == 'y':
                stus.pop(del_i)
                print('学生已被删除！')
            else:
                print('您已取消该操作！')

        else:
            print('该学生不存在，请重新输入！')
    elif user_choose == 4:
        print('谢谢使用！再见！')
        input('按回车键退出！')
        break

    else:
        print('您的输入有误，请重新输入！')
    print('-' * 62)
