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
        self.resnet = tv.models.resnet50(pretrained=True)
        # self.conv1 = nn.Conv2d(config.input_channel,64,7,2,3,bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        # self.final_pool = torch.nn.MaxPool2d(3, 2)

    def forward(self,x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # print(x.shape)
        x = self.final_pool(x).squeeze()
        # x = x.flatten(start_dim=1)
        return x


class LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=2000):
        super(LSTM, self).__init__()
        # features = 18432
        features = 2048
        layers = 2
        output = 4
        self.LSTM = nn.LSTM(features, lstm_hidden_size, layers, batch_first=True)
        self.Linear = nn.Linear(lstm_hidden_size, output)

    def forward(self, x):
        out, (h_n, c_n) = self.LSTM(x)
        out_last = out[:,-1,:]
        out_last = h_n[-1,:, :]
        out_last = self.Linear(out_last)
        return out_last


# class ResNetLSTM(nn.Module):
#     def __init__(self,lstm_hidden_size=2000):
#         super(ResNetLSTM,self).__init__()
#         self.LSTM = LSTM()
#         self.get_feature = ExtractFeature()

#     def forward(self,x):
#         for one_pic in range(x.size(0)):
#             one_pic_feature_extracted = self.get_feature(x[one_pic])
            



if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from dataloader.dataset import TreatmentRrequirement
    model1 = ExtractFeature().cuda()
    model2 = LSTM().cuda()
    train_dataset = TreatmentRrequirement(config.data_path, transform='train')
    criteria = nn.BCELoss()
    optimizer = torch.optim.Adam(model1.parameters(), config.lr)

    img_extract = torch.zeros(len(train_dataset), config.seq_len,18432)
    with torch.no_grad():
        for i in range(len(train_dataset)):
            input = train_dataset[i][0]
            input = input.cuda()
            output = model1(input)
            img_extract[i] = output
            