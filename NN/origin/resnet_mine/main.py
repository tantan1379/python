#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/11/12 21:18
from resnet import *
import torch
import torchvision.datasets
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from resnet import *
from tqdm import tqdm


# Load model and data
def load(dev):
    mod = resnet34().to(dev)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss().to(dev)
    data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    train_set = torchvision.datasets.CIFAR10(root='../../Database/Cifar-10', train=True, download=False,
                                             transform=data_transform["train"])
    validate_set = torchvision.datasets.CIFAR10(root='../../Database/Cifar-10', train=False, download=False,
                                                transform=data_transform["val"])
    # 得到数据集的迭代器(train_loader.shape=torch.size(150,200,3,32,32)
    tl = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True)
    vl = torch.utils.data.DataLoader(validate_set, batch_size=200, shuffle=False)
    return mod, opt, crit, tl, vl


# train and validate for the net
def run(epochs, model, optimizer, criterion, train_loader, validate_loader, device):
    for epoch in range(epochs):
        # train
        model.train()
        best_acc = 0.0
        for train_index, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
            rate = (train_index + 1) / len(train_loader)
            print("\rtrain loss: {:.3f}".format(train_loss))


if __name__ == "__main__":
    device = torch.device("cuda")
    epochs = 2

    model, optimizer, criterion, train_loader, validate_loader = load(device)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, 10)
    model.to(device)
    run(epochs, model, optimizer, criterion, train_loader, validate_loader, device)
