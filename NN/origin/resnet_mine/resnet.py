#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: resnet.py
# author: twh
# time: 2020/11/12 21:18
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np


# Use for ResNet 18/34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # downsample is a function

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)  # relu after add
        out += identity  # add up direct and shortcut path output
        out = self.relu(out)

        return out


# Use for ResNet 50/101/152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=self.expansion * out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channel)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # block表示使用的残差结构块的类型(basic/bottleneck)
    # blocks_num表示残差结构的个数（以列表形式写入）
    # num_classes表示分类数
    # include_top
    def __init__(self, block, blocks_num, num_classes=1000):
        super(ResNet, self).__init__()
        # inplanes = depth of input
        self.inplanes = 128
        # shape of input = (3x112x112) # RGB three channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer实际上以块为单位对层进行操作
        self.layer1 = self._make_layer(block, 64, blocks_num[0], 1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], 2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], 2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    # block表示使用的残差结构块的类型(basic/bottleneck)
    # planes表示某残差结构中第一层对应卷积核的个数
    # blocks_num表示残差结构的个数（以列表形式写入）
    def _make_layer(self, block, planes, blocks_num, stride=1):
        downsample = None
        # 第1层
        # 18/34 不执行下列语句；
        # 50/101/152 执行语句；conv2调整深度
        # 2-5层
        # 任何层的resnet都执行
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 生成下采样函数（用于shortcut中identity的计算）
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=block.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * planes)
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        # 表示下一层需要输入的卷积层深度的变化
        self.inplanes = planes * block.expansion
        # 构建重复的实线层(从第二个层开始，因为第一层（虚线）已构建完毕）
        for _ in range(1, blocks_num):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output


def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


