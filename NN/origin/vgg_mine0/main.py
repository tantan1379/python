#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/11/11 18:54
import argparse
from net import *

if __name__ == '__main__':
    # 保证可复现性
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # 设置句柄
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'])
    parser.add_argument('--epochs', default='4')
    parser.add_argument('--opt', default='Adam', choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--lr', default='0.001')
    args = parser.parse_args()
    # 根据句柄进行配置
    num_epochs = int(args.epochs)
    lr = float(args.lr)
    opt = args.opt
    if args.device == 'GPU':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # 对模型进行训练和测试
    model, train_loader, test_loader, loss_func = load(device)
    optimizer = opt_choose(opt,lr,model)
    train_best_acc = train(num_epochs, model, loss_func, optimizer, train_loader, device)
    test_acc = test(model, test_loader, device)
    # 显示训练和测试结果
    print('train accuracy of the model on the train images: {} %'
          .format(train_best_acc))
    print('Test accuracy of the model on the test images: {} %'
          .format(test_acc))
