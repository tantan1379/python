import sys
sys.path.append("..")
from config import config
from torch.utils.data import DataLoader
import os
import time
import torch
import torchvision as tv
import torch.nn as nn
import numpy as np

class ExtractFeature(nn.Module):
    def __init__(self):
        super(ExtractFeature, self).__init__()
        self.conv1 = nn.Conv2d(16,64,7,2,3,bias=False)
        self.resnet = tv.models.resnet50(pretrained=True)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        # self.final_pool = torch.nn.MaxPool2d(3, 2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.final_pool(x).squeeze()
        # x = x.flatten(start_dim=1)
        return x

class LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=2000):
        super(LSTM, self).__init__()
        # features = 18432
        features = 2048
        layers = 2
        output = 1
        self.LSTM = nn.LSTM(features, lstm_hidden_size, layers, batch_first=True)
        self.Linear = nn.Linear(lstm_hidden_size, output)

    def forward(self, x):
        out, _ = self.LSTM(x)
        out_last = out[:, -1, :]
        out_last = self.Linear(out_last)
        return out_last

