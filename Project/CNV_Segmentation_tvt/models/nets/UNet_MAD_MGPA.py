import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.layers import unetConv2,unetUp,unetConv2_dilation
from models.utils.init_weights import init_weights
import math
from torchvision import models
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

try:
    from deformable_conv import DeformConv2d
except:
    from .deformable_conv import DeformConv2d


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

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
class Dilat_Chnnel_Atten(nn.Module):
    def __init__(self, inplanes):
        super(Dilat_Chnnel_Atten, self).__init__()
        self.dilate1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True)
        )

        self.dilate2 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True)
        )
        self.dilate3 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
        )

        self.dilate4 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=7, dilation=7, bias=False),
            nn.BatchNorm2d(inplanes // 4),
            nn.ReLU(inplace=True),
        )

        self.channel_att = SELayer(channel=inplanes)

        self.deform = nn.Sequential(
            DeformConv2d(inplanes, inplanes, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.spatial = SpatialGate()
        self.gamma = torch.nn.Parameter(torch.Tensor([1.0]))
        self.beta = torch.nn.Parameter(torch.Tensor([1.0]))


    def forward(self, x):
        residual = x
        x_1 = self.dilate1(x)
        x_2 = self.dilate2(x)
        x_3 = self.dilate3(x)
        x_4 = self.dilate4(x)

        x_att_in = self.deform(torch.cat([x_1, x_2, x_3, x_4], dim=1))
        x_CH_Att = x_att_in * (self.channel_att(x_att_in))
        x_SP_Att = x_att_in * (self.spatial(x_att_in))

        residual = self.relu(residual + self.gamma * x_CH_Att + self.beta * x_SP_Att)

        return residual

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

class SPPblock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[1, 1], stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,bias=False),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True)
        )
        self.conv_smoonth = nn.Sequential(
                    nn.Conv2d(in_channels=4*in_channels+out_channels, out_channels=out_channels, kernel_size=1,bias=False),
                    # nn.BatchNorm2d(out_channels),
                    # nn.ReLU(inplace=True)
                )
        self.se1=SELayer1(4*in_channels+out_channels)

    def forward(self, x_up,x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x_up)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x_up)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x_up)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x_up)), size=(h, w), mode='bilinear')


        out = torch.cat([self.layer1,self.layer2, self.layer3, self.layer4, x], 1)
        out1 = self.se1(out)
        out1 = self.conv_smoonth(out1)

        return out1




class UNet_MGPA_MAD(nn.Module):

    def __init__(self, in_channels=1,n_classes=4,feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet_MGPA_MAD, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.dmsa = Dilat_Chnnel_Atten(inplanes=512)


        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.spp1 = SPPblock(filters[0],filters[1])
        self.spp2 = SPPblock(filters[1],filters[2])
        self.spp3 = SPPblock(filters[2],filters[3])
        self.fd1 = nn.Conv2d(filters[0]*6, filters[0], 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)       # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)     # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)     # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)     # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)   # 256*32*64
        center = self.dmsa(center)
        print(conv3.shape)
        print(conv4.shape)
        conv4_sp=self.spp3(conv3,conv4)
        conv3_sp = self.spp2(conv2, conv3)
        conv2_sp = self.spp1(conv1, conv2)



        up4 = self.up_concat4(center,conv4_sp)  # 128*64*128
        up3 = self.up_concat3(up4,conv3_sp)     # 64*128*256
        up2 = self.up_concat2(up3,conv2_sp)     # 32*256*512
        up1 = self.up_concat1(up2,conv1)     # 16*512*1024

        final_1 = self.final_1(up1)

        #return F.softmax(final_1,dim=1)
        return torch.sigmoid(final_1)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = UNet_MGPA(in_channels=1, n_classes=1, is_deconv=True).cuda()
    print(net)
    x = torch.rand((4, 1, 256, 128)).cuda()
    forward = net.forward(x)
    #print(forward)
    print(type(forward))