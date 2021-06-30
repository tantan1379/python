import torch.nn as nn
import math
import torch
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import models
from functools import partial
from torch.nn import init

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

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class DSMA_cpf(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(DSMA_cpf, self).__init__()

        filters = [64, 128, 256, 512]
        expan = [128, 256, 512]
        spatial_ch = [64, 64]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.mce_2 = GPG_2([spatial_ch[-1], expan[0], expan[1], expan[2]], width=spatial_ch[-1], up_kwargs=up_kwargs)
        self.mce_3 = GPG_3([expan[0], expan[1], expan[2]], width=expan[0], up_kwargs=up_kwargs)
        self.mce_4 = GPG_4([expan[1], expan[2]], width=expan[1], up_kwargs=up_kwargs)

        self.dmsa = Dilat_Chnnel_Atten(inplanes=512)

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
        e4 = self.encoder4(e3)
        m2 = self.mce_2(e1, e2, e3, e4)     #64
        m3 = self.mce_3(e2, e3, e4)          #128
        m4 = self.mce_4(e3, e4)           #256
        # d_bottom=self.bottom(c5)

        # Center

        e4=self.dmsa(e4)

        # Decoder
        d4 = self.decoder4(e4) + m4
        a1 = F.upsample(d4, size=size, mode='bilinear')
        d3 = self.decoder3(d4) + m3
        a2 = F.upsample(d3, size=size, mode='bilinear')
        d2 = self.decoder2(d3) + m2
        a3 = F.upsample(d2, size=size, mode='bilinear')
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = torch.sigmoid(out)
        print(out.shape)

        return out

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        print(out.shape)
        print('----------')
        return out

class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        self.dac = DACblock(width)

        # self.dilation1 = nn.Sequential(
        #     SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        # self.dilation2 = nn.Sequential(
        #     SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        # self.dilation3 = nn.Sequential(
        #     SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        # feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
        feat = self.conv_out(feat)
        feat=self.dac(feat)
        return feat


class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        self.dac = DACblock(width)

        # self.dilation1 = nn.Sequential(
        #     SeparableConv2d(2 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        # self.dilation2 = nn.Sequential(
        #     SeparableConv2d(2 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        # feat = torch.cat([self.dilation1(feat), self.dilation2(feat)], dim=1)
        feat = self.conv_out(feat)
        feat= self.dac(feat)
        return feat


class GPG_2(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

        self.dac=DACblock(width)

        # self.dilation1 = nn.Sequential(
        #     SeparableConv2d(4 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        # self.dilation2 = nn.Sequential(
        #     SeparableConv2d(4 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        # self.dilation3 = nn.Sequential(
        #     SeparableConv2d(4 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        # self.dilation4 = nn.Sequential(
        #     SeparableConv2d(4 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        print(feat.shape)
        # feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
        #                  dim=1)
        feat = self.conv_out(feat)
        feat = self.dac(feat)
        print((feat.shape))
        # feat = self.conv_out(feat)
        return feat


if __name__ == "__main__":
    import torch
    import numpy as np

    print('begin...')

    # model = Encoder_Path(64)
    model = DSMA_cpf(num_classes=1).cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    tmp = torch.randn(4, 3, 256, 256).cuda()
    y = torch.randn(1, 448, 448)

    import time

    start_time = time.time()
    print(model(tmp).shape)
    end_time = time.time()
    print("Time ==== {}".format(end_time - start_time))
    print('done')
