#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: unet.py.py
# author: twh
# time: 2020/12/14 13:23
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, 1, 1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(self.init_weights)

    def forward(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)


class InConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(InConv, self).__init__()
        self.conv = DoubleConv(inchannel, outchannel)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inchannel, outchannel)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, inchannel, outchannel, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(inchannel, inchannel // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(inchannel, outchannel)
        self.up.apply(self.init_weight)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffy = x2.size()[2]-x1.size[2]
        diffx = x2.size()[3]-x1.size[3]

        x1
