#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/11/10 20:37
import torch
import vgg
import os,sys
import matplotlib.pyplot as plt
# import argparse


if __name__ == '__main__':
    test_result = 0
    epochs = 5
    accuracy = 0
    choice = 0
    choices = ['Adam', 'SGD', 'RMSprop']
    approach = 'cuda'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cuda', default='GPU', choices=['GPU', 'CPU'], help='use GPU?')
    # arg = parser.parse_args()
    # if arg.cuda == 'GPU':
    device = torch.device(approach)
    # elif arg.cuda == 'CPU':
    #     device = torch.device('cpu')

    model, train_loader, test_loader, loss_func, optimizer = vgg.load(device)

    while True:
        print('*' * 19 + ' Configuration ' + '*' * 20)
        print(' ' * 15 + ' device={} epochs={} '.format(approach, epochs))
        print(' ' * 14 + ' optimization method: {} '.format(choices[choice]))
        print('*' * 54)
        s = int(input('please input the mode:\n1.train 2.analysis 3.change epochs 4.select opt 5.exit\n'))
        while s != 1 and s != 2 and s != 3 and s != 4 and s != 5:
            s = int(input('Invalid input!Please input the right number again!\n1.train 2.analysis 3.change epochs '
                          '4.select opt 5.exit\n'))

        if s == 1:
            for epoch in range(1, epochs + 1):
                print('\n===> epoch: %d/%d' % (epoch, epochs))
                train_result,Loss_list, Accuracy_list = vgg.train(model, train_loader, loss_func, optimizer, device)
                print('*****the accuracy of training set is: %.2f%%' % train_result)
                test_result = vgg.test(model, test_loader, loss_func, device)
                print('*****the accuracy of test set is: %.2f%%' % test_result)
                accuracy = max(accuracy, test_result)
            print('Train finished! Select analysis option to see the result!')
            print('\n')

        elif s == 2:
            if accuracy == 0:
                print('No data exist! Please train first!')
            else:
                print('(In %d epochs, the highest accuracy of test set is: %.2f%%' % accuracy)
            print('\n')

        elif s == 3:
            epochs = int(input("please input the number of epochs:"))
            print('\n')

        elif s == 4:
            choice = int(input("please choose the optimization method:1.Adam 2.SGD 3.RMSprop:")) - 1
            vgg.opt_choice(model, choice)
            print('\n')

        elif s == 5:
            python = sys.executable
            os.execl(python, python, *sys.argv)
