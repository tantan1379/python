#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: vgg.py
# author: twh
# time: 2020/11/10 12:37
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np


# VGG11 structure:
# (1)conv-relu-pool
# (2)conv-relu-pool
# (3)conv-relu-conv-relu-pool
# (4)conv-relu-conv-relu-pool
# (5)conv-relu-conv-relu-pool
# (6)fc


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # the first layer(input_channels=3,output_channels=64)
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # the second layer(input_channels=64,output_channels=128)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # the third layer(input_channels=128,output_channels=256)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # the fourth layer(input_channels=256,output_channels=512)
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # the fifth layer(input_channels=512,output_channels=512)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifer = nn.Sequential(
            # the sixth layer(fc)
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x


def opt_choice(model, choice):
    if choice == 0:
        return optim.Adam(model.parameters(), lr=0.001)
    elif choice == 1:
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif choice == 2:
        return optim.RMSprop(model.parameters(), lr=0.001)


def train(model, train_loader, loss_func, optimizer, device):
    print('train:')
    Loss_list = []
    Accuracy_list = []
    train_loss = 0
    train_correct = 0
    train_acc=0
    total = 0
    i = 0
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 每次优化时先将梯度清零
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # torch.max(input,1) 取每行的最大值，返回max_values和indexs两个tensor
        prediction = torch.max(output, 1)
        total += len(target)
        train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        train_acc = 100 * train_correct / total
        if i % 100 == 99:
            avg_loss = train_loss / 100
            Loss_list.append(avg_loss)
            Accuracy_list.append(train_acc)
    return train_acc, Loss_list, Accuracy_list


def test(model, test_loader, loss_func, device):
    print('test:')
    test_loss = 0
    test_correct = 0
    total = 0
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_func(output, target)
        test_loss += loss.item()
        prediction = torch.max(output, 1)
        total += len(target)
        test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
    return 100 * test_correct / total


def load(device):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    # Load dataset
    train_set = datasets.CIFAR10(root=r"F:/Database/Cifar-10", train=True, download=False, transform=train_transform)
    test_set = datasets.CIFAR10(root=r"F:/Database/Cifar-10", train=False, download=False, transform=test_transform)
    # turn the dataset into iterable object, with 200 batches
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=200, shuffle=False)
    # environment configuration
    model = VGG().to(device)
    torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss().to(device)
    return model, train_loader, test_loader, loss_func, optimizer
