import torch
from torch import nn
from torchvision.models import resnet34
from torch.nn import init
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,middle_channels,kernel_size=1,bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(middle_channels),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1,bias=False),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)


    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        x = self.conv_3x3(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear',
                              align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output


class ResUNet_PP(nn.Module):
    def __init__(self, num_classes):

        super(ResUNet_PP, self).__init__()
        nb_filter = [64, 64, 128, 256, 512]

        self.up0_1 = DecoderBlock(nb_filter[1],nb_filter[0],relu=False) #128
        self.up1_1 = DecoderBlock(nb_filter[2],nb_filter[1],relu=False) #128
        self.up0_2 = DecoderBlock(nb_filter[1],nb_filter[0],relu=False) #128
        self.up2_1 = DecoderBlock(nb_filter[3],nb_filter[2],relu=False) #128
        self.up1_2 = DecoderBlock(nb_filter[2],nb_filter[1],relu=False) #128
        self.up0_3 = DecoderBlock(nb_filter[1],nb_filter[0],relu=False) #128
        self.up3_1 = DecoderBlock(nb_filter[4],nb_filter[3],relu=False) #128
        self.up2_2 = DecoderBlock(nb_filter[3],nb_filter[2],relu=False) #128
        self.up1_3 = DecoderBlock(nb_filter[2],nb_filter[1],relu=False) #128
        self.up0_4 = DecoderBlock(nb_filter[1],nb_filter[0],relu=False) #128
        self.up0_4 = DecoderBlock(nb_filter[1],nb_filter[0],relu=False) #128

        self.Params_0 = nn.Parameter(torch.ones(6)*0.0)
        self.Params_1 = nn.Parameter(torch.ones(3)*0.0)
        self.Params_2 = nn.Parameter(torch.ones(1)*0.0)

        self.backbone = resnet34(pretrained=True)



        self.conv0_0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu
        )
        self.conv1_0 = nn.Sequential(
            self.backbone.maxpool,
            self.backbone.layer1
        )

        self.conv2_0 = self.backbone.layer2

        self.conv3_0 = self.backbone.layer3

        self.conv4_0 = self.backbone.layer4


        self.conv0_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])

        self.conv1_1 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])

        self.final = BaseNetHead(nb_filter[0], num_classes, 2,
                             is_aux=False, norm_layer=nn.BatchNorm2d)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(x0_0+self.up0_1(x1_0))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(x1_0+self.up1_1(x2_0))
        x0_2 = self.conv0_2(self.Params_0[0]*x0_0+x0_1+self.up0_2(x1_1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(x2_0+self.up2_1(x3_0))
        x1_2 = self.conv1_2(self.Params_1[0]*x1_0+x1_1+self.up1_2(x2_1))
        x0_3 = self.conv0_3(self.Params_0[1]*x0_0+self.Params_0[2]*x0_1+x0_2+self.up0_3(x1_2))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(x3_0+self.up3_1(x4_0))
        x2_2 = self.conv2_2(self.Params_2[0]*x2_0+x2_1+self.up2_2(x3_1))
        x1_3 = self.conv1_3(self.Params_1[1]*x1_0+self.Params_1[2]*x1_1+x1_2+self.up1_3(x2_2))
        x0_4 = self.conv0_4(self.Params_0[3]*x0_0+self.Params_0[4]*x0_1+self.Params_0[5]*x0_2+x0_3+self.up0_4(x1_3))

        output = self.final(x0_4)
        return torch.sigmoid(output)#,self.Params_0,self.Params_1,self.Params_2


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = ResUNet_PP  (num_classes=1)
    import numpy as np
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    model = model.cuda(0)
    print(model(images)[0].size())
