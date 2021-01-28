#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: densenet.py
# author: twh
# time: 2020/12/22 9:35
from torch import nn
import torch


# conv block layer
def _conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer


class DensBlock(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(DensBlock, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(_conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


# transition layer
def _transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer


class DenseNet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(DenseNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )

        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(DensBlock(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(_transition(channels, channels // 2))  # 通过transition 层将大小减半，通道数减半
                channels = channels // 2

        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))

        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
