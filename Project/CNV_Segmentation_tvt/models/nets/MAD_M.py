import torch.nn as nn
import math
import torch
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
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


class MAD_M(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(MAD_M, self).__init__()

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

        self.dmsa = Dilat_Chnnel_Atten(inplanes=512)
        # self.squeeze4 = nn.Sequential(nn.Conv2d(filters[2], filters[1], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[1]), nn.ReLU(inplace=True))
        # self.bnre4=nn.Sequential(nn.BatchNorm2d(filters[1]), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(filters[2], filters[1], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[1]), nn.ReLU(inplace=True))
        self.bnre3 = nn.Sequential(nn.BatchNorm2d(filters[1]), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(filters[1], filters[0], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True))
        self.bnre2 = nn.Sequential(nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True))
        # self.squeeze1 = nn.Sequential(nn.Conv2d(filters[1], filters[0], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True))
        # self.bnre1 = nn.Sequential(nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True))

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
        # print(self.firstconv)
        size=x.shape[2:]
        x1 = self.firstconv(x)
        x2 = self.firstbn(x1)
        x3 = self.firstrelu(x2)
        x4 = self.firstmaxpool(x3)
        e1 = self.encoder1(x4)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)   #512,8,8

        # Center


        e4=self.dmsa(e4)  #512,8,8

        # Decoder
        md4 = self.decoder4(e4)   #256,16,16


        d3 = self.decoder3(md4)   #128,32,32
        a1 = F.upsample(md4, size=d3.shape[2:], mode='bilinear')
        md3 = self.squeeze3(a1)   #128,32,32
        mul3= torch.mul(md3,e2)
        result3 = self.bnre3(torch.add(mul3,d3))



        d2 = self.decoder2(result3)
        a2 = F.upsample(result3, size=d2.shape[2:], mode='bilinear')
        md2 = self.squeeze2(a2)
        mul2 = torch.mul(md2,e1)
        result2 = self.bnre2(torch.add(mul2,d2))
        # a3 = F.upsample(d2, size=size, mode='bilinear')
        d1 = self.decoder1(result2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = torch.sigmoid(out)
        print(out.shape)

        return out

if __name__ == "__main__":
    import torch
    import numpy as np

    print('begin...')

    # model = Encoder_Path(64)
    model = MAD_M(num_classes=1).cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    tmp = torch.randn(2, 3, 256, 256).cuda()
    y = torch.randn(1, 448, 448)

    import time

    start_time = time.time()
    print(model(tmp).shape)
    end_time = time.time()
    print("Time ==== {}".format(end_time - start_time))
    print('done')
