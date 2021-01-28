#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: resnet.py
# author: twh
# time: 2020/11/16 21:41
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.conv3 = nn.Conv2d(out_channel,)