import torch.nn as nn
import math
import torch
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import models
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)


    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return scale

class SELayer1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, bias=False)


    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = self.conv1x1(torch.cat([y_avg, y_max], 1)).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SPPblock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SPPblock, self).__init__()

        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,bias=False),
        )
        self.conv_smoonth = nn.Sequential(
                    nn.Conv2d(in_channels=3*in_channels+out_channels, out_channels=out_channels, kernel_size=1,bias=False),
                )
        self.se1=SELayer1(3*in_channels+out_channels)

    def forward(self, x_up,x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.layer2 = F.upsample(self.conv(self.pool2(x_up)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x_up)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x_up)), size=(h, w), mode='bilinear')


        out = torch.cat([self.layer2, self.layer3, self.layer4, x], 1)
        out1 = self.se1(out)
        out1 = self.conv_smoonth(out1)

        return out1

class SPPResUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(SPPResUNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.spp1 = SPPblock(filters[0],filters[0])
        self.spp2 = SPPblock(filters[0],filters[1])
        self.spp3 = SPPblock(filters[1],filters[2])
        self.spp4 = SPPblock(filters[2],filters[3])
        self.fd1 = nn.Conv2d(filters[0]*5, filters[0], 1)



        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.firstconv(x)
        x2 = self.firstbn(x1)
        e0 = self.firstrelu(x2)



        x4 = self.firstmaxpool(e0)  #64,64,64
        e1 = self.encoder1(x4)    #64 ,64, 64


        e2 = self.encoder2(e1)    #128,32,32


        e3 = self.encoder3(e2)    #256,16,16

        e4 = self.encoder4(e3)    #512,8,8

        e1_sp = self.spp1(e0,e1)
        e2_sp = self.spp2(e1,e2)
        e3_sp = self.spp3(e2,e3)
        e4_sp = self.spp4(e3,e4)


        # Decoder
        d4 = self.decoder4(e4_sp) + e3_sp    #256
        d3 = self.decoder3(d4) + e2_sp     #128

        d2 = self.decoder2(d3) + e1_sp   #64,64,64
        d1 = self.decoder1(d2) + e0       #64,128,128
        out = self.finaldeconv1(d1)

        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = torch.sigmoid(out)


        return out

if __name__ == "__main__":
    import torch
    import numpy as np

    print('begin...')

    # model = Encoder_Path(64)
    model = SPPResUNet(num_classes=1).cuda()
    # print(model)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    tmp = torch.randn(2, 3, 512, 512).cuda()
    y = torch.randn(1, 448, 448)

    import time

    start_time = time.time()
    print(model(tmp).shape)
    end_time = time.time()
    print("Time ==== {}".format(end_time - start_time))
    print('done')
