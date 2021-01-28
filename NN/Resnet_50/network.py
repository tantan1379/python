#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: network.py
# author: twh
# time: 2020/11/5 14:45
import torch
import torch.nn as nn
from torch.autograd import Variable


class convolutional_block(nn.Module):  # convolutional_block层
    def __init__(self, cn_input, cn_middle, cn_output, s=2):
        super(convolutional_block, self).__init__()
        self.step1 = nn.Sequential(nn.Conv2d(cn_input, cn_middle, (1, 1), (s, s), padding=0, bias=False),
                                   nn.BatchNorm2d(cn_middle, affine=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(cn_middle, cn_middle, (3, 3), (1, 1), padding=(1, 1), bias=False),
                                   nn.BatchNorm2d(cn_middle, affine=False), nn.ReLU(inplace=True),
                                   nn.Conv2d(cn_middle, cn_output, (1, 1), (1, 1), padding=0, bias=False),
                                   nn.BatchNorm2d(cn_output, affine=False))
        self.step2 = nn.Sequential(nn.Conv2d(cn_input, cn_output, (1, 1), (s, s), padding=0, bias=False),
                                   nn.BatchNorm2d(cn_output, affine=False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_tmp = x
        x = self.step1(x)
        x_tmp = self.step2(x_tmp)
        x = x + x_tmp
        x = self.relu(x)
        return x


class identity_block(nn.Module):  # identity_block层
    def __init__(self, cn, cn_middle):
        super(identity_block, self).__init__()
        self.step = nn.Sequential(nn.Conv2d(cn, cn_middle, (1, 1), (1, 1), padding=0, bias=False),
                                  nn.BatchNorm2d(cn_middle, affine=False), nn.ReLU(inplace=True),
                                  nn.Conv2d(cn_middle, cn_middle, (3, 3), (1, 1), padding=1, bias=False),
                                  nn.BatchNorm2d(cn_middle, affine=False), nn.ReLU(inplace=True),
                                  nn.Conv2d(cn_middle, cn, (1, 1), (1, 1), padding=0, bias=False),
                                  nn.BatchNorm2d(cn, affine=False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_tmp = x
        x = self.step(x)
        x = x + x_tmp
        x = self.relu(x)
        return x


class Resnet(nn.Module):  # 主层
    def __init__(self, c_block, i_block):
        super(Resnet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 64, (7, 7), (2, 2), padding=(3, 3), bias=False),
                                  nn.BatchNorm2d(64, affine=False), nn.ReLU(inplace=True), nn.MaxPool2d((3, 3), 2, 1))
        self.layer1 = c_block(64, 64, 256, 1)
        self.layer2 = i_block(256, 64)
        self.layer3 = c_block(256, 128, 512)
        self.layer4 = i_block(512, 128)
        self.layer5 = c_block(512, 256, 1024)
        self.layer6 = i_block(1024, 256)
        self.layer7 = c_block(1024, 512, 2048)
        self.layer8 = i_block(2048, 512)
        self.out = nn.Linear(2048, 2, bias=False)
        self.avgpool = nn.AvgPool2d(7, 7)

    def forward(self, input):
        x = self.conv(input)
        x = self.layer1(x)
        for i in range(2):
            x = self.layer2(x)
        x = self.layer3(x)
        for i in range(3):
            x = self.layer4(x)
        x = self.layer5(x)
        for i in range(5):
            x = self.layer6(x)
        x = self.layer7(x)
        for i in range(2):
            x = self.layer8(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


net = Resnet(convolutional_block, identity_block).cuda()