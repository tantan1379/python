#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: net.py
# author: twh
# time: 2020/11/11 18:54
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import time


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64, affine=True),
            # nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # layer 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            # nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # layer 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            # nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # layer 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            # nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # layer 5
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            # nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def opt_choose(opt, lr, model):
    if opt == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif opt == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        print('error!')
        return None


def train(num_epochs, model, loss_func, optimizer, train_loader, device):
    train_best_acc = 0
    print('Start training:')
    for epoch in range(num_epochs):
        train_correct = 0
        total = 0
        time.sleep(0.1)
        for batch_num, (data, target) in tqdm(enumerate(train_loader), desc='进行中'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            train_loss = loss.item()
            # 反向传播得到所有关于loss的梯度属性
            loss.backward()
            # 更新梯度
            optimizer.step()
            prediction = torch.max(output, 1)
            total += len(target)
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            train_acc = train_correct / total * 100
            train_best_acc = max(train_acc, train_best_acc)
            if batch_num % 50 == 49:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.5f}'
                      .format(epoch + 1, num_epochs, batch_num + 1, len(train_loader), train_loss))
                print('Acc: {:.2f}%'.format(train_acc))
    print('Training finished!')
    return train_best_acc


def test(model, test_loader, device):
    print('Start test:')
    test_correct = 0
    total = 0
    test_acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        prediction = torch.max(output, 1)
        total += len(target)
        test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        test_acc += test_correct / total * 100
    return test_acc/len(test_loader)   # len(test_loader)=len(test_dataset)/batch_size


def load(device):
    # 数据集预处理
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='F:/Database/Cifar-10', train=True, download=False,
                                             transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='F:/Database/Cifar-10', train=False, download=False,
                                            transform=test_transform)
    # 得到数据集的迭代器(train_loader.shape=torch.size(150,200,3,32,32)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=False)
    # 环境配置
    model = VGG16().to(device)
    loss_func = nn.CrossEntropyLoss().to(device)

    return model, train_loader, test_loader, loss_func
