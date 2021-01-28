#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/11/5 14:45
import numpy as np
import random
import os
import torch
import torchvision
from torchvision import datasets, transforms
import network
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import time


def data_load(dir, batch, a=0, b=0, c=0):  # 读取图片文件
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(a),  # 旋转，增加训练样本数量
        transforms.RandomHorizontalFlip(b),  # 水平翻转，增加训练样本数量
        transforms.RandomVerticalFlip(c),  # 垂直翻转，增加训练样本数量
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    return train_loader


def train(optimizer):
    loss_function = nn.CrossEntropyLoss().cuda()
    time1 = time.time()
    epco = 2
    batch = 16
    miss = 0
    add = 0
    loss_all = 0
    loss_down = 0
    loss_add = 0
    train_pic = 0
    train_loader = data_load(dir=r"F:\Database\dog&cat\train", batch=16, a=90, b=0.5, c=0.5)
    if train_pic > 0:
        print(train_pic)
    else:
        print(str(len(train_loader) * epco))
    for e in range(epco):
        z = []
        x = []
        for i, (images, labels) in enumerate(train_loader):
            x = Variable(images.cuda())
            z = Variable(labels.cuda())
            out = network.net.forward(x)
            loss = loss_function(out, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            out = network.net.forward(x)
            loss1 = loss_function(out, z)
            loss_all += loss.item()
            loss_down += (loss.item() - loss1.item())
            if loss1 > loss:
                loss_add = (loss1.item() - loss.item())
                add += 1
            if loss >= 0.5 / batch:
                print(loss1)
                print("====" + str(loss))
                miss += 1
                print("----------------------------------------------------------")
            if (i + 1) % 100 == 0:
                if train_pic > 0:
                    print(str((e * train_pic + i + 1) / (epco * train_pic) * 100) + "%")
                else:
                    print(str((e * len(train_loader) + i + 1) / (epco * len(train_loader)) * 100) + "%")
            if i % 3000 == 0 and i != 0:
                state = {"net": network.net.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(state, "net.pth")
            if 0 < train_pic <= i:
                break
    # 下面是一些监控输出，可以不用管
    print("add=" + str(add))
    if train_pic > 0:
        print("acc=" + str((epco * train_pic - miss) / (epco * train_pic) * 100) + "%")
        print("avgloss=" + str(loss_all / (epco * train_pic)))
        print("avgloss_down=" + str(loss_down / (epco * train_pic)))
    else:
        print("acc=" + str((epco * len(train_loader) - miss) / (epco * len(train_loader)) * 100) + "%")
        print("avgloss=" + str(loss_all / (epco * len(train_loader))))
        print("avgloss_down=" + str(loss_down / (epco * len(train_loader))))
    if add > 0:
        print("avgloss_add=" + str(loss_add / add))
    print(str(time.time() - time1) + "秒")
    # 保存参数和模型
    state = {"net": network.net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, "net.pth")


def test():  # 单张测试
    train_loader = data_load(dir=r"F:\Database\dog&cat\test", batch=1)
    for i, (images, labels) in enumerate(train_loader):
        x = Variable(images.cuda())
        z = Variable(labels.cuda())
        out = network.net.forward(x)
        print(z)
        print(out)
        break
    if out.argmax() == z:
        print("正确")
    else:
        print("错误")


def acc(s):  # 多张测试，分成训练集和验证集
    if s == 0:
        dir = "train"
        type = "训练集"
    else:
        dir = "test"
        type = "验证集"
    train_loader = data_load(dir, batch=1)
    corret = 0
    all = 300
    if all > len(train_loader):
        all = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        x = Variable(images.cuda())
        z = Variable(labels.cuda())
        out = network.net.forward(x)
        if out.argmax() == z:
            corret += 1
        if i >= all:
            break
    print(type + "正确率:===============================================================================" + str(
        corret / all * 100) + "%")


while True:
    rate = 0.0001
    optimizer = torch.optim.Adam(network.net.parameters())
    if not os.path.exists("net.pth"):
        state = {"net": network.net.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, "net.pth")
    else:
        checkpoint = torch.load("net.pth")
        network.net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer = torch.optim.Adam(network.net.parameters(), rate)
    s = input("1:训练   2:分析   3:正确率(300张)   0:退出:")
    while s != "1" and s != "2" and s != "3" and s != "0":
        s = input("输入错误，重新输入：1:训练   2:分析   3:正确率(300张)   0:退出:")
    if s == "1":
        train(optimizer)
    elif s == "2":
        test()
    elif s == "3":
        acc(0)
        acc(1)
    elif s == "0":
        break
